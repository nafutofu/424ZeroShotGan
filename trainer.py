# Cleaned-up trainer.py for grayscale-to-color image translation (ZstGAN adaptation)

import os
import time
import datetime
import random
import torch
import torch.nn.functional as F
import itertools
from torchvision.utils import save_image
from model import ContentEncoder, Decoder, Discriminator, Generator
import cv2
import numpy as np
import clip

import torchvision.models as models
import torch.nn as nn

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15], resize=True):
        super().__init__()
        vgg = models.vgg16(weights="VGG16_Weights.DEFAULT").features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.selected_layers = layers
        self.resize = resize
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_vgg, y_vgg = x, y
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x_vgg = layer(x_vgg)
            y_vgg = layer(y_vgg)
            if i in self.selected_layers:
                loss += self.criterion(x_vgg, y_vgg)
        return loss

class Solver(object):
    def __init__(self, data_loader, config):
        self.config = config
        self.data_loader = data_loader


        # Model and training configs
        self.ft_num = config.ft_num
        self.image_size = config.image_size
        self.c_dim = config.c_dim
        self.d_conv_dim = config.d_conv_dim
        self.d_repeat_num = config.d_repeat_num
        self.lambda_rec = config.lambda_rec
        self.lambda_hist = config.lambda_hist
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.n_critic = config.n_critic
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.test_iters = config.test_iters
        self.num_test_imgs = config.num_test_imgs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        if self.device.type == 'cuda':
             print(torch.cuda.get_device_name(0))
        print()
        self.vgg_loss = VGGPerceptualLoss().to(self.device)

        # Output directories
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Training schedule
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        self.build_model()

    def lab_to_rgb(self, L, ab):
        # De-normalise
        L = L * 50 + 50
        ab = ab * 127

        lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)

        rgb_images = []
        for img_lab in lab:
            img_lab = img_lab.astype(np.float32)
            img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_Lab2RGB)
            img_rgb = np.clip(img_rgb, 0, 1)  # ensure valid range
            rgb_images.append(torch.from_numpy(img_rgb).permute(2, 0, 1))  # (3, H, W)

        return torch.stack(rgb_images).to(self.device)

    def compute_soft_histogram_loss(self, pred_ab, gt_ab, bins=64, range_min=-1.0, range_max=1.0, sigma=0.02):

        device = pred_ab.device
        bin_centers = torch.linspace(range_min, range_max, steps=bins).to(device)  # [bins]

        loss = 0.0

        for channel in range(2):  # a and b channels
            pred_vals = pred_ab[:, channel, :, :].reshape(-1)  # [B*H*W]
            gt_vals = gt_ab[:, channel, :, :].reshape(-1)

            # [bins, B*H*W]
            pred_diff = pred_vals[None, :] - bin_centers[:, None]  # broadcasting
            gt_diff = gt_vals[None, :] - bin_centers[:, None]

            pred_weights = torch.exp(-0.5 * (pred_diff / sigma) ** 2)  # Gaussian kernel
            gt_weights = torch.exp(-0.5 * (gt_diff / sigma) ** 2)

            # [bins]
            pred_hist = pred_weights.sum(dim=1)
            gt_hist = gt_weights.sum(dim=1)

            # Normalise to sum to 1
            pred_hist /= (pred_hist.sum() + 1e-6)
            gt_hist /= (gt_hist.sum() + 1e-6)

            loss += F.l1_loss(pred_hist, gt_hist)
        return loss


    def build_model(self):
        # Define networks
        self.G = Generator(
            input_dim=1,
            output_dim=2,
            dim=64,
            n_res=8,
            n_downsample=2,
            n_upsample=2,
            norm_type='in',
            res_norm='in',
            activ='relu',
            pad_type='reflect',
            clip_dim=512,
            image_size=self.image_size
        ).to(self.device)

        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num, ft_num=self.ft_num)

        # Define optimizers
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(),
            self.g_lr, [self.beta1, self.beta2]
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), self.d_lr, [self.beta1, self.beta2]
        )
        self.D.to(self.device)

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()


    def restore_model(self, resume_iters):
        print(f'Loading model from iteration {resume_iters}')
        self.G.load_state_dict(torch.load(os.path.join(self.model_save_dir, f'{resume_iters}-G.ckpt')))
        self.D.load_state_dict(torch.load(os.path.join(self.model_save_dir, f'{resume_iters}-D.ckpt')))


    def train(self):
        print('Start training...')
        start_time = time.time()

        g_lr = self.g_lr
        d_lr = self.d_lr

        if self.resume_iters:
            self.restore_model(self.resume_iters)

        self._train_iter = iter(self.data_loader.train)

        for i in range(self.num_iters):
            try:
                L, ab, prompt_embed = next(self._train_iter)
            except (AttributeError, StopIteration):
                self._train_iter = iter(self.data_loader.train)
                L, ab, prompt_embed = next(self._train_iter)

            L = L.to(self.device)
            ab = ab.to(self.device)
            prompt_embed = prompt_embed.to(self.device, dtype=torch.float32)


            # === Train Discriminator ===
            fake_ab = self.G(L, prompt_embed)
            fake_lab = torch.cat([L, fake_ab], dim=1)
            real_lab = torch.cat([L, ab], dim=1)

            out_real, _ = self.D(real_lab)
            d_loss_real = -torch.mean(out_real)

            out_fake, _ = self.D(fake_lab.detach())
            d_loss_fake = torch.mean(out_fake)

            d_loss = d_loss_real + d_loss_fake

            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # === Train Generator ===
            if (i + 1) % self.n_critic == 0:
                fake_ab = self.G(L, prompt_embed)
                fake_lab = torch.cat([L, fake_ab], dim=1)
                out_fake, _ = self.D(fake_lab)

                g_loss_fake = -torch.mean(out_fake)
                g_loss_rec = F.l1_loss(fake_ab, ab)
                g_loss_hist = self.compute_soft_histogram_loss(fake_ab, ab)
                g_loss = (
                    g_loss_fake +
                    self.lambda_rec * g_loss_rec +
                    self.lambda_hist * g_loss_hist
                )

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

            # === Logging ===
            if (i + 1) % self.log_step == 0:
                elapsed = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
                log = f"Elapsed [{elapsed}], Iteration [{i + 1}/{self.num_iters}]\n"
                log += f"D/loss_real: {d_loss_real.item():.4f}, D/loss_fake: {d_loss_fake.item():.4f}\n"
                log += f"G/loss_fake: {g_loss_fake.item():.4f}, G/loss_rec: {g_loss_rec.item():.4f}, G/loss_hist: {g_loss_hist.item():.4f}"
                print(log)

            # === Save Sample Images ===
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    fake_ab = self.G(L, prompt_embed)
                    fake_rgb = self.lab_to_rgb(L, fake_ab)
                    real_rgb = self.lab_to_rgb(L, ab)
                    gray3 = L.expand(-1, 3, -1, -1)
                    merged = torch.cat([gray3, fake_rgb, real_rgb], dim=0)
                    sample_path = os.path.join(self.sample_dir, f'{i + 1}_samples.jpg')
                    save_image(merged.data.cpu(), sample_path, nrow=L.size(0))
                    print(f'Saved samples to {sample_path}')

            # === Save Model Checkpoints ===
            if (i + 1) % self.model_save_step == 0:
                torch.save(self.G.state_dict(), os.path.join(self.model_save_dir, f'{i + 1}-G.ckpt'))
                torch.save(self.D.state_dict(), os.path.join(self.model_save_dir, f'{i + 1}-D.ckpt'))
                print(f'Saved model checkpoints at iteration {i + 1}')

            # === Decay Learning Rates ===
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.config.num_iters_decay):
                g_lr -= self.g_lr / float(self.config.num_iters_decay)
                d_lr -= self.d_lr / float(self.config.num_iters_decay)
                for param_group in self.g_optimizer.param_groups:
                    param_group['lr'] = g_lr
                for param_group in self.d_optimizer.param_groups:
                    param_group['lr'] = d_lr
                print(f'Decayed learning rates, g_lr: {g_lr:.6f}, d_lr: {d_lr:.6f}')



    def test(self, prompt="red flowers"):
        from torchvision.utils import save_image

        print(f"[INFO] Running test with prompt: '{prompt}'")

        # Load CLIP
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

        with torch.no_grad():
            # Encode the prompt to CLIP embedding
            token = clip.tokenize([prompt]).to(self.device)
            prompt_embed = self.clip_model.encode_text(token)  # [1, 512]
            prompt_embed = prompt_embed.to(dtype=torch.float32)

            # Loop through test data
            for idx, (L, ab, _) in enumerate(self.data_loader.test):
                L = L.to(self.device)               # [1, 1, H, W]
                prompt_embed_batch = prompt_embed.repeat(L.size(0), 1)  # match batch size

                fake_ab = self.G(L, prompt_embed_batch)  # colourise

                fake_rgb = self.lab_to_rgb(L, fake_ab)
                gray3 = L.expand(-1, 3, -1, -1)
                merged = torch.cat([gray3, fake_rgb], dim=0)

                # Save image
                prompt_tag = prompt.replace(" ", "_").lower()
                os.makedirs(self.result_dir, exist_ok=True)
                out_path = os.path.join(self.result_dir, f"{idx+1}_{prompt_tag}.jpg")
                save_image(merged.cpu(), out_path, nrow=gray3.size(0))
                print(f"[✓] Saved: {out_path}")




