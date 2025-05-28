### sampling.py
import torch
import numpy as np
import einops
import imageio

@torch.no_grad()
def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=100, gif_name="sampling.gif", c=1, h=28, w=28):
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    if device is None:
        device = ddpm.device

    x = torch.randn(n_samples, c, h, w).to(device)

    for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
        time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
        eta_theta = ddpm.backward(x, time_tensor)

        alpha_t = ddpm.alphas[t]
        alpha_t_bar = ddpm.alpha_bars[t]

        x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

        if t > 0:
            z = torch.randn(n_samples, c, h, w).to(device)
            beta_t = ddpm.betas[t]
            sigma_t = beta_t.sqrt()
            x = x + sigma_t * z

        if idx in frame_idxs or t == 0:
            normalized = x.clone()
            for i in range(len(normalized)):
                normalized[i] -= torch.min(normalized[i])
                normalized[i] *= 255 / torch.max(normalized[i])

            frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
            frame = frame.cpu().numpy().astype(np.uint8)
            frames.append(frame)

    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            rgb_frame = np.repeat(frame, 3, axis=2)
            writer.append_data(rgb_frame)
            if idx == len(frames) - 1:
                for _ in range(frames_per_gif // 3):
                    writer.append_data(rgb_frame)
    return x