import torch.nn as nn
import torch
from ddpm.utils import sinusoidal_embedding

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
        self.bottleneck = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
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




class TinyUNet2(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super().__init__()

        # 时间嵌入（不参与训练）
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # 时间嵌入投影成不同阶段的通道数
        self.temb_proj1 = nn.Linear(time_emb_dim, 32)
        self.temb_proj2 = nn.Linear(time_emb_dim, 64)
        self.temb_proj3 = nn.Linear(time_emb_dim, 128)

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.SiLU()
        )
        self.down1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # → 14x14
        self.enc2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU()
        )
        self.down2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # → 7x7

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
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

        # 获取时间嵌入向量
        t_emb = self.time_embed(t)

        # Encoder
        x1 = self.enc1(x) + self.temb_proj1(t_emb).view(n, -1, 1, 1)  # (n, 32, 28, 28)
        x2 = self.enc2(self.down1(x1)) + self.temb_proj2(t_emb).view(n, -1, 1, 1)  # (n, 64, 14, 14)
        x3 = self.bottleneck(self.down2(x2)) + self.temb_proj3(t_emb).view(n, -1, 1, 1)  # (n, 128, 7, 7)

        # Decoder
        u1 = self.up1(x3)  # (n, 64, 14, 14)
        u1 = self.dec1(torch.cat([u1, x2], dim=1))

        u2 = self.up2(u1)  # (n, 32, 28, 28)
        u2 = self.dec2(torch.cat([u2, x1], dim=1))

        out = self.out(u2)  # (n, 1, 28, 28)
        return out



def get_group_norm(num_channels):
    for g in [8, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )

        self.block = nn.Sequential(
            get_group_norm(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            get_group_norm(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        temb = self.time_proj(t_emb).view(x.size(0), -1, 1, 1)
        out = self.block(x) + temb
        return out + self.res_conv(x)

class TinyResUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=128):
        super().__init__()

        # 时间嵌入：sinusoidal embedding + MLP
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.temb_net = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder
        self.enc1 = ResBlock(1, 32, time_emb_dim)
        self.down1 = nn.Conv2d(32, 32, 4, stride=2, padding=1)  # 28→14
        self.enc2 = ResBlock(32, 64, time_emb_dim)
        self.down2 = nn.Conv2d(64, 64, 4, stride=2, padding=1)  # 14→7

        # Bottleneck
        self.bot = ResBlock(64, 128, time_emb_dim)

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 7→14
        self.dec1 = ResBlock(128, 64, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 14→28
        self.dec2 = ResBlock(64, 32, time_emb_dim)

        self.out = nn.Sequential(
            get_group_norm(32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.temb_net(self.time_embed(t))

        # Encoder
        x1 = self.enc1(x, t_emb)
        x2 = self.enc2(self.down1(x1), t_emb)
        x3 = self.bot(self.down2(x2), t_emb)

        # Decoder
        u1 = self.up1(x3)
        u1 = self.dec1(torch.cat([u1, x2], dim=1), t_emb)

        u2 = self.up2(u1)
        u2 = self.dec2(torch.cat([u2, x1], dim=1), t_emb)

        return self.out(u2)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.view(B, C, H * W)  # → (B, C, N)

        q = self.q(x_flat)
        k = self.k(x_flat)
        v = self.v(x_flat)

        attn = torch.softmax((q.transpose(1, 2) @ k) / (C ** 0.5), dim=-1)  # (B, N, N)
        out = v @ attn.transpose(1, 2)  # (B, C, N)
        out = self.proj(out)

        out = out.view(B, C, H, W)
        return x + out  # Residual connection


class TinyResUNetWithAttention(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=128):
        super().__init__()

        # 时间嵌入：sinusoidal embedding + MLP
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.temb_net = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder
        self.enc1 = ResBlock(1, 32, time_emb_dim)
        self.down1 = nn.Conv2d(32, 32, 4, stride=2, padding=1)  # 28→14
        self.enc2 = ResBlock(32, 64, time_emb_dim)
        self.down2 = nn.Conv2d(64, 64, 4, stride=2, padding=1)  # 14→7

        # Bottleneck
        self.bot = ResBlock(64, 128, time_emb_dim)
        self.bot_attn = AttentionBlock(128)
        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 7→14
        self.dec1 = ResBlock(128, 64, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 14→28
        self.dec2 = ResBlock(64, 32, time_emb_dim)

        self.out = nn.Sequential(
            get_group_norm(32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.temb_net(self.time_embed(t))

        # Encoder
        x1 = self.enc1(x, t_emb)
        x2 = self.enc2(self.down1(x1), t_emb)
        x3 = self.bot(self.down2(x2), t_emb)
        x3 = self.bot_attn(x3)

        # Decoder
        u1 = self.up1(x3)
        u1 = self.dec1(torch.cat([u1, x2], dim=1), t_emb)

        u2 = self.up2(u1)
        u2 = self.dec2(torch.cat([u2, x1], dim=1), t_emb)

        return self.out(u2)