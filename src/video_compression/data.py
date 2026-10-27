from pathlib import Path


def image_paths(root: Path) -> list[Path]:
    paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not paths:
        raise ValueError(f"No PNG frames in {root}")
    return paths


def load_image(path, size, device):
    # Match baseline preprocessing, including Pillow's default RGB resize filter.
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as source:
        original_size = source.size
        image = source.convert("RGB")
        if image.size != size:
            image = image.resize(size)
        values = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return tensor.to(device), original_size
