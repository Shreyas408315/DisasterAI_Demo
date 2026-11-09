
from pathlib import Path
import csv

ROOT = Path("data/sen1floods11/v1.1")
OUT = Path("data")

IMAGE_SOURCE = ROOT / "data/flood_events/HandLabeled/S1Hand"
MASK_SOURCE = ROOT / "data/flood_events/HandLabeled/LabelHand"
SPLIT_SOURCE = ROOT / "splits/flood_handlabeled"

IMAGE_OUT = OUT / "images"
MASK_OUT = OUT / "masks"
SPLIT_OUT = OUT / "splits"

for folder in (IMAGE_OUT, MASK_OUT, SPLIT_OUT):
    folder.mkdir(parents=True, exist_ok=True)

# Link the original TIFFs instead of copying them.
def link_or_copy(source, destination):
    if destination.exists():
        return
    try:
        destination.symlink_to(source.resolve())
    except (OSError, NotImplementedError):
        # Fall back to copying only if Windows cannot create a symlink.
        import shutil
        shutil.copy2(source, destination)

split_files = {
    "train": SPLIT_SOURCE / "flood_train_data.csv",
    "val": SPLIT_SOURCE / "flood_valid_data.csv",
    "test": SPLIT_SOURCE / "flood_test_data.csv",
}

split_names = {"train": [], "val": [], "test": []}
seen = set()

for split, csv_path in split_files.items():
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing official split: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue

            image_name, mask_name = row[0].strip(), row[1].strip()
            image_path = IMAGE_SOURCE / image_name
            mask_path = MASK_SOURCE / mask_name

            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing pair in {split}: {image_path} | {mask_path}"
                )

            key = Path(image_name).stem
            if key in seen:
                raise ValueError(f"Duplicate image across splits: {key}")
            seen.add(key)

            # Matching stems let the existing discover_pairs() loader pair them.
            link_or_copy(image_path, IMAGE_OUT / image_name)
            link_or_copy(mask_path, MASK_OUT / mask_name)
            split_names[split].append(key)

for split, names in split_names.items():
    (SPLIT_OUT / f"{split}.txt").write_text(
        "\n".join(names) + "\n", encoding="utf-8"
    )
    print(f"{split}: {len(names)} official image-mask pairs")

print(f"\nPrepared dataset under: {OUT.resolve()}")
print("Original Sen1Floods11 files remain unchanged.")
