from pathlib import Path
import numpy as np
from PIL import Image
import rasterio

def read_raster(path, image_size=None, is_mask=False):
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            arr = src.read()
    else:
        arr = np.array(Image.open(path))
        if arr.ndim == 2:
            arr = arr[None, ...]
        else:
            arr = np.transpose(arr, (2, 0, 1))

    if is_mask:
        arr = arr[0]
        # Keep original integer label values; negative values are ignored by metric/loss.
        if image_size:
            im = Image.fromarray(arr.astype(np.float32))
            im = im.resize((image_size, image_size), Image.Resampling.NEAREST)
            arr = np.array(im)
        return arr.astype(np.int64)

    # Keep first two bands for Sentinel-1 VV/VH; duplicate single-band input.
    if arr.shape[0] >= 2:
        arr = arr[:2]
    else:
        arr = np.repeat(arr[:1], 2, axis=0)

    arr = arr.astype(np.float32)
    # Expected Sentinel-1 dB convention; inspect your files before relying on this.
    arr = np.nan_to_num(arr, nan=-50.0, posinf=1.0, neginf=-50.0)
    arr = np.clip(arr, -50.0, 1.0)
    arr = (arr + 50.0) / 51.0

    if image_size:
        resized = []
        for band in arr:
            im = Image.fromarray(band.astype(np.float32))
            resized.append(np.array(im.resize((image_size, image_size), Image.Resampling.BILINEAR)))
        arr = np.stack(resized)
    return arr.astype(np.float32)

def synthetic_pair(index, size=128, seed=0):
    """Artificial toy pair for smoke testing only; not a real disaster dataset."""
    rng = np.random.default_rng(seed + index)
    yy, xx = np.mgrid[:size, :size]
    center = size * (0.35 + 0.3 * rng.random())
    width = size * (0.08 + 0.07 * rng.random())
    water = ((xx - center) ** 2 / (width ** 2) + (yy - size * 0.5) ** 2 / ((size * .42) ** 2)) < 1
    water = water.astype(np.int64)
    band1 = rng.normal(0.55, 0.12, (size, size)).astype(np.float32)
    band2 = rng.normal(0.55, 0.12, (size, size)).astype(np.float32)
    band1[water == 1] = rng.normal(0.18, 0.05, (water == 1).sum())
    band2[water == 1] = rng.normal(0.22, 0.05, (water == 1).sum())
    return np.clip(np.stack([band1, band2]), 0, 1), water

def discover_pairs(data_root):
    root = Path(data_root)
    image_dir, mask_dir = root / "images", root / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Expected {image_dir} and {mask_dir}. See README.md for dataset setup."
        )
    valid = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"}
    images = {
    p.stem.removesuffix("_S1Hand"): p
    for p in image_dir.rglob("*")
    if p.suffix.lower() in valid
    }
    masks = {
    p.stem.removesuffix("_LabelHand"): p
    for p in mask_dir.rglob("*")
    if p.suffix.lower() in valid
    }
    keys = sorted(set(images) & set(masks))
    if not keys:
        raise RuntimeError("No image/mask pairs with matching filename stems were found.")
    split_dir = root / "splits"
    split_map = {}
    if split_dir.exists():
        for split in ("train", "val", "test"):
            f = split_dir / f"{split}.txt"
            if f.exists():
                split_map[split] = { Path(x.strip()).stem.removesuffix("_S1Hand") 
                    for x in f.read_text(encoding="utf-8").splitlines()
                    if x.strip() and not x.startswith("#")
                }
    pairs = []
    for k in keys:
        split = next((s for s, names in split_map.items() if k in names), None)
        pairs.append((k, images[k], masks[k], split))
    if split_map:
        missing = [k for k, _, _, s in pairs if s is None]
        if missing:
            print(f"Warning: {len(missing)} paired files are not in split files; excluding them.")
        pairs = [p for p in pairs if p[3] is not None]
    else:
        # Deterministic fallback split; event-aware official splits are strongly preferred.
        rng = np.random.default_rng(42)
        order = np.arange(len(pairs))
        rng.shuffle(order)
        n = len(order)
        labels = {}
        for j, idx in enumerate(order):
            labels[idx] = "train" if j < int(.7*n) else ("val" if j < int(.85*n) else "test")
        pairs = [(k, im, ma, labels[i]) for i, (k, im, ma, _) in enumerate(pairs)]
        print("WARNING: no split files found; using random chip split. This may leak event/geographic information.")
    return {s: [(k, im, ma) for k, im, ma, sp in pairs if sp == s] for s in ("train", "val", "test")}
