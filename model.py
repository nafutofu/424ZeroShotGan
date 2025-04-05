import torch
import torch.nn as nn
import torch.nn.functional as F

def adain(content_feat, style_feat):
    # content_feat: (B, C, H, W), style_feat: (B, C)
    size = content_feat.size()
    style_mean = style_feat.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
    style_std = torch.ones_like(style_mean)  # Default scale of 1

    content_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    content_std = content_feat.std(dim=[2, 3], keepdim=True) + 1e-5

    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean


# === Content Encoder ===
class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        layers = [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for _ in range(n_downsample):
            layers += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        layers += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*layers)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

# === Decoder ===
class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='in', activ='relu', pad_type='reflect'):
        super(Decoder, self).__init__()
        layers = [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        for _ in range(n_upsample):
            layers += [nn.Upsample(scale_factor=2),
                       Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        layers += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*layers)

    def forward(self, x, style_mean=None, style_std=None):
        for i, layer in enumerate(self.model):
            if isinstance(layer, ResBlocks) and style_mean is not None and style_std is not None:
                x = adain(layer(x), style_mean, style_std)
            else:
                x = layer(x)
        return x

class Generator(nn.Module):
    def __init__(
        self,
        input_dim=1,
        output_dim=2,
        dim=64,
        n_res=8,
        n_downsample=2,
        n_upsample=2,
        norm_type="in",
        res_norm="in",
        activ="relu",
        pad_type="reflect",
        clip_dim=512,
        image_size=128
    ):
        super().__init__()
        self.encoder = ContentEncoder(
            n_downsample=n_downsample,
            n_res=n_res,
            input_dim=input_dim,
            dim=dim,
            norm=norm_type,
            activ=activ,
            pad_type=pad_type
        )

        self.decoder = Decoder(
            n_upsample=n_upsample,
            n_res=n_res,
            dim=self.encoder.output_dim,
            output_dim=output_dim,
            res_norm=res_norm,
            activ=activ,
            pad_type=pad_type
        )

        # Style network for AdaIN
        self.style_mlp = nn.Sequential(
            nn.Linear(clip_dim, self.encoder.output_dim),
            nn.ReLU(),
            nn.Linear(self.encoder.output_dim, self.encoder.output_dim)
        )

    def forward(self, L, prompt_embed):
        content_feat = self.encoder(L)                    # [B, C, H, W]
        style_vector = self.style_mlp(prompt_embed)       # [B, C]
        fake_ab = self.decoder(content_feat, style_vector)
        return fake_ab



# === Residual Blocks ===
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='reflect'):
        super(ResBlocks, self).__init__()
        self.model = nn.Sequential(*[ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type) for _ in range(num_blocks)])

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='reflect'):
        super(ResBlock, self).__init__()
        layers = [
            Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type),
            Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.model(x)

# === Conv2dBlock ===
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='reflect'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True

        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            raise ValueError(f"Unsupported padding type: {pad_type}")

        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        else:
            self.norm = None

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

# === LayerNorm ===
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

# === Discriminator ===
class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=1, repeat_num=6, ft_num=16):
        super(Discriminator, self).__init__()
        layers = [nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.01)]
        curr_dim = conv_dim
        for _ in range(1, repeat_num):
            layers += [nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.01)]
            curr_dim *= 2

        kernel_size = int(image_size / (2 ** repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, ft_num, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
