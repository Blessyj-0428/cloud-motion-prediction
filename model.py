import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)

        self.dec3 = CBR(256 + 128, 128)
        self.dec2 = CBR(128 + 64, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d3 = self.up(e3)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        out = self.final(d2)
        return out


class DDPM(nn.Module):
    def __init__(self, denoise_model, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = denoise_model
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward(self, x_t, t):
        return self.model(x_t)

    def sample(self, shape, device):
        x = torch.randn(shape).to(device)
        for t in reversed(range(self.num_timesteps)):
            noise_pred = self.model(x)
            alpha = self.alphas[t].to(device)
            beta = self.betas[t].to(device)

            x = (1 / alpha.sqrt()) * (
                x - beta / (1 - self.alphas_cumprod[t]).sqrt() * noise_pred
            )

            if t > 0:
                noise = torch.randn_like(x)
                x += beta.sqrt() * noise

        return x
