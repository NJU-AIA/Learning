
import torch.nn as nn
import torch
from ddpm.utils import sinusoidal_embedding

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class ResNetForDDPM(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super().__init__()

        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, 28 * 28),
            nn.SiLU()
        )

        self.input_proj = nn.Conv2d(2, 32, kernel_size=3, padding=1)

        self.resblocks = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.output_proj = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        n = x.size(0)
        t = t.view(-1)
        t_emb = self.time_proj(self.time_embed(t))  # (n, 784)
        t_feat = t_emb.view(n, 1, 28, 28)  # → feature map
        x = torch.cat([x, t_feat], dim=1)  # (n, 2, 28, 28)

        h = self.input_proj(x)
        h = self.resblocks(h)
        out = self.output_proj(h)
        return out