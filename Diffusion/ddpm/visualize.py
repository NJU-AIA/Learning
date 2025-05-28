### visualize.py
import matplotlib.pyplot as plt
import torch

def show_images(images, title=""):
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** 0.5)
    cols = round(len(images) / rows)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(images):
                break
            fig.add_subplot(rows, cols, idx + 1)
            plt.imshow(images[idx][0], cmap="gray")
            plt.axis("off")
            idx += 1
    fig.suptitle(title, fontsize=20)
    plt.show()

def show_forward(ddpm, loader, device):
    for batch in loader:
        imgs = batch[0]
        show_images(imgs, "Original images")
        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(imgs.to(device),
                     [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]),
                f"DDPM Noisy images {int(percent * 100)}%"
            )
        break