
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src_disasterai import discover_pairs
from run_experiment import FloodDataset

splits = discover_pairs("data")
print("Official split counts:", {k: len(v) for k, v in splits.items()})

dataset = FloodDataset(splits["train"][:8], size=64)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

for i, (x, y) in enumerate(loader):
    print(f"\nBatch {i + 1}")
    print("Image shape:", tuple(x.shape))
    print("Image min/max/mean:", float(x.min()), float(x.max()), float(x.mean()))
    print("Image finite:", bool(x.isfinite().all()))
    print("Mask labels:", np.unique(y.numpy(), return_counts=True))
    print("Labeled pixels:", int((y >= 0).sum()))
    print("Flood fraction among labeled pixels:",
          float((y == 1).sum() / (y >= 0).sum().clamp(min=1)))
