# Cleaned-up trainer.py for grayscale-to-color image translation (ZstGAN adaptation)

import os
import time
import datetime
import random
import torch
import torch.nn.functional as F
import itertools
from torchvision.utils import save_image
from model import ContentEncoder, Decoder, Discriminator

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

    def build_model(self):
        # Define networks
        self.encoder = ContentEncoder(
            n_downsample=2,
            n_res=8,
            input_dim=1,
            dim=64,
            norm='in',
            activ='relu',
            pad_type='reflect'
        )
        self.decoder = Decoder(
            n_upsample=2,
            n_res=8,
            dim=256,           # dim should match encoder's output
            output_dim=3,      # RGB output
            res_norm='in',
            activ='relu',
            pad_type='reflect'
        )

        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num, ft_num=self.ft_num)

        # Define optimizers
        self.g_optimizer = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
            self.g_lr, [self.beta1, self.beta2]
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), self.d_lr, [self.beta1, self.beta2]
        )

        # Move models to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.D.to(self.device)

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        return (x + 1) / 2

    def restore_model(self, resume_iters):
        print(f'Loading model from iteration {resume_iters}')
        self.encoder.load_state_dict(torch.load(os.path.join(self.model_save_dir, f'{resume_iters}-encoder.ckpt')))
        self.decoder.load_state_dict(torch.load(os.path.join(self.model_save_dir, f'{resume_iters}-decoder.ckpt')))
        self.D.load_state_dict(torch.load(os.path.join(self.model_save_dir, f'{resume_iters}-D.ckpt')))

    def train(self):
        print('Start training...')
        start_time = time.time()

        g_lr = self.g_lr
        d_lr = self.d_lr

        if self.resume_iters:
            self.restore_model(self.resume_iters)

        for i in range(self.num_iters):
            try:
                gray, rgb = next(self._train_iter)
            except (AttributeError, StopIteration):
                self._train_iter = iter(self.data_loader.train)
                gray, rgb = next(self._train_iter)

            gray = gray.to(self.device)
            rgb = rgb.to(self.device)

            # === Train Discriminator ===
            out_src_real, _ = self.D(rgb)
            d_loss_real = -torch.mean(out_src_real)

            x_content = self.encoder(gray)
            batch_size = gray.size(0)
            x_fake = self.decoder(x_content)

            out_src_fake, _ = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src_fake)

            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # === Train Generator ===
            if (i + 1) % self.n_critic == 0:
                out_src_fake, _ = self.D(x_fake)
                g_loss_fake = -torch.mean(out_src_fake)
               #  g_loss_rec = F.l1_loss(x_fake, rgb)
                x_fake_vgg = (x_fake + 1) / 2
                rgb_vgg = (rgb + 1) / 2
                g_loss_rec = self.vgg_loss(x_fake_vgg, rgb_vgg)
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

            # === Logging ===
            if (i + 1) % self.log_step == 0:
                et = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
                log = f"Elapsed [{et}], Iteration [{i + 1}/{self.num_iters}]"
                log += f", D/loss_real: {d_loss_real.item():.4f}, D/loss_fake: {d_loss_fake.item():.4f}"
                if (i + 1) % self.n_critic == 0:
                    log += f", G/loss_fake: {g_loss_fake.item():.4f}, G/loss_rec: {g_loss_rec.item():.4f}"
                print(log)

            # === Save Sample Images ===
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    merged = torch.cat([gray.expand(-1, 3, -1, -1), x_fake, rgb], dim=0)
                    sample_path = os.path.join(self.sample_dir, f'{i + 1}_samples.jpg')
                    save_image(self.denorm(merged.data.cpu()), sample_path, nrow=gray.size(0))
                    print(f'Saved samples to {sample_path}')

            # === Save Model Checkpoints ===
            if (i + 1) % self.model_save_step == 0:
                torch.save(self.encoder.state_dict(), os.path.join(self.model_save_dir, f'{i + 1}-encoder.ckpt'))
                torch.save(self.decoder.state_dict(), os.path.join(self.model_save_dir, f'{i + 1}-decoder.ckpt'))
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



    def test(self):
        print('Running test...')
        self.restore_model(self.test_iters)
        self.encoder.eval()
        self.decoder.eval()

        # Get total test size
        test_len = len(self.data_loader.test)
        max_visuals = self.config.num_test_imgs

        # Pick N unique indices randomly
        visual_indices = set(random.sample(range(test_len), max_visuals))
        test_batch = 1

        with torch.no_grad():
            for i, (gray, rgb) in enumerate(self.data_loader.test):
                gray = gray.to(self.device)
                rgb = rgb.to(self.device)
                content = self.encoder(gray)
                fake = self.decoder(content)

                if i in visual_indices:

                    merged = torch.cat([gray.expand(-1, 3, -1, -1), fake, rgb], dim=0)
                    result_path = os.path.join(self.result_dir, f'{test_batch + 1}_test.jpg')
                    save_image(self.denorm(merged.data.cpu()), result_path, nrow=gray.size(0))
                    print(f'Saved result to {result_path}')
                    test_batch += 1

                # METRICS HERE


