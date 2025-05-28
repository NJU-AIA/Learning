# models/conv.py
import torch.nn as nn
import torch
from ddpm.utils import sinusoidal_embedding

class SimpleConvNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super().__init__()
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.time_proj = nn.Sequential(nn.Linear(time_emb_dim, 28 * 28), nn.SiLU())

        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, 1, 1)
        )

    def forward(self, x, t):
        n = x.size(0)
        t_feat = self.time_proj(self.time_embed(t)).view(n, 1, 28, 28)
        x = torch.cat([x, t_feat], dim=1)
        return self.net(x)
