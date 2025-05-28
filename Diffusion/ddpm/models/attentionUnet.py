import torch.nn as nn
import torch
from ddpm.utils import sinusoidal_embedding

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.out = nn.Conv2d(in_channels, in_channels, 1)
        self.scale = in_channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).reshape(B, C, H * W)
        k = self.key(x).reshape(B, C, H * W)
        v = self.value(x).reshape(B, C, H * W)

        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale  # (B, HW, HW)
        attn = attn.softmax(dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1)).reshape(B, C, H, W)
        return self.out(out + x)  # residual connection


class TinyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super().__init__()

        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, 28 * 28),
            nn.SiLU()
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.SiLU()
        )
        self.down1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # → (14x14)
        self.enc2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU()
        )
        self.down2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # → (7x7)

        # Bottleneck
        # self.bottleneck = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Conv2d(128, 128, 3, padding=1),
        #     nn.SiLU()
        # )
        self.bottleneck = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
            SelfAttention(128),  # 👈 Attention Block
            nn.SiLU()
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # → 14x14
        self.dec1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, padding=1)
        )
        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # → 28x28
        self.dec2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1)
        )

        self.out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, t):
        n = x.size(0)
        t = t.view(-1)
        t_emb = self.time_proj(self.time_embed(t)).view(n, 1, 28, 28)
        x = torch.cat([x, t_emb], dim=1)  # (n, 2, 28, 28)

        # Encoder
        x1 = self.enc1(x)  # (n, 32, 28, 28)
        x2 = self.enc2(self.down1(x1))  # (n, 64, 14, 14)
        x3 = self.bottleneck(self.down2(x2))  # (n, 128, 7, 7)

        # Decoder
        u1 = self.up1(x3)  # (n, 64, 14, 14)
        u1 = self.dec1(torch.cat([u1, x2], dim=1))  # skip from x2

        u2 = self.up2(u1)  # (n, 32, 28, 28)
        u2 = self.dec2(torch.cat([u2, x1], dim=1))  # skip from x1

        out = self.out(u2)  # (n, 1, 28, 28)
        return out
