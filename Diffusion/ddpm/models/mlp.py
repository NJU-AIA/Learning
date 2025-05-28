# models/mlp.py
import torch.nn as nn
import torch
from ddpm.utils import sinusoidal_embedding

class SimpleMLP(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100, image_size=28*28):
        super().__init__()
        self.image_size = image_size
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.input_proj = nn.Linear(image_size + time_emb_dim, 64)
        self.hidden = nn.Sequential(
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU()
        )
        self.output_proj = nn.Linear(64, image_size)

    def forward(self, x, t):
        n = x.size(0)
        x = x.view(n, -1)
        t_emb = self.time_embed(t.view(-1))
        x = torch.cat([x, t_emb], dim=1)
        h = self.input_proj(x)
        h = self.hidden(h)
        return self.output_proj(h).view(n, 1, 28, 28)
