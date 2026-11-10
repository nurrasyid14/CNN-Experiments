from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def export_binary_tensor(image_path: Path, processed_dir: Path, image_size: int = 224) -> str:
    """Generate normalized C++ tensor in CNNTNS1 format."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    name_hash = abs(hash(image_path.name))
    tensor_name = f"{name_hash}.bin"
    tensor_path = processed_dir / tensor_name
    if tensor_path.is_file() and tensor_path.stat().st_size > 0:
        return str(tensor_path.resolve())

    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    chw = np.ascontiguousarray(norm.transpose(2, 0, 1), dtype=np.float32)

    magic = b"CNNTNS1\0"
    dims = np.array([3, image_size, image_size], dtype="<u4")
    with tensor_path.open("wb") as out:
        out.write(magic)
        out.write(dims.tobytes())
        out.write(chw.tobytes())
    return str(tensor_path.resolve())


def load_classes(path: Path) -> tuple[list[str], dict[tuple[int, int, int], str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        rows = csv.DictReader(source)
        rows = list(rows)
    names = [row["name"].strip() for row in rows]
    colors = {
        (int(row[" r"]), int(row[" g"]), int(row[" b"])): row["name"].strip()
        for row in rows
    }
    return names, colors


def prepare_simplicity(root: Path, config: dict, image_size: int = 224) -> tuple[list[dict], Counter]:
    raw_dir = resolve(root, config["raw_dir"])
    divisor = int(config["label_divisor"])
    processed_dir = resolve(root, f"data/processed/simplicity")
    records = []
    counts = Counter()

    for split in config["split_names"]:
        split_dir = raw_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing simplicity split directory: {split_dir}")
        for image_path in sorted(split_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = int(image_path.stem) // divisor
            tensor_path = export_binary_tensor(image_path, processed_dir, image_size)
            records.append({
                "path": str(image_path.resolve()),
                "label": label,
                "split": split,
                "tensor_path": tensor_path,
            })
            counts[f"{split}/class_{label}"] += 1

    write_jsonl(resolve(root, config["manifest"]), records)
    return records, counts


def prepare_aerial(root: Path, config: dict, image_size: int = 224) -> tuple[list[dict], Counter]:
    raw_dir = resolve(root, config["raw_dir"])
    images_dir = raw_dir / config["images_dir"]
    masks_dir = raw_dir / config["masks_dir"]
    class_names, colors = load_classes(resolve(root, config["class_csv"]))
    processed_dir = resolve(root, f"data/processed/aerial")
    counts = Counter()
    samples = []

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mask_path = masks_dir / (image_path.stem + ".png")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing aerial mask: {mask_path}")

        mask_image = Image.open(mask_path)
        mask = np.asarray(mask_image)
        if mask_image.mode in {"1", "L", "I", "I;16"} or mask.ndim == 2:
            known = Counter(int(index) for index in mask.flat)
            class_counts = {
                class_names[index]: count
                for index, count in known.items()
                if 0 <= index < len(class_names) and class_names[index] != "unlabeled"
            }
        else:
            pixels = mask.reshape(-1, 3)
            known = Counter((int(pixel[0]), int(pixel[1]), int(pixel[2])) for pixel in pixels)
            class_counts = {
                colors[color]: count
                for color, count in known.items()
                if color in colors and colors[color] != "unlabeled"
            }
        if not class_counts:
            raise ValueError(f"No documented semantic class found in {mask_path}")

        label = max(class_counts, key=class_counts.get)
        samples.append((str(image_path.resolve()), str(mask_path.resolve()), label))
        counts[label] += 1

    split_config = config["split"]
    rng = random.Random(int(split_config["seed"]))
    rng.shuffle(samples)

    by_class = {}
    for sample in samples:
        by_class.setdefault(sample[2], []).append(sample)

    records = []
    for label, class_samples in by_class.items():
        rng.shuffle(class_samples)
        train_count = int(len(class_samples) * float(split_config["train"]))
        validation_count = int(len(class_samples) * float(split_config["validation"]))
        for index, (image_path, mask_path, _) in enumerate(class_samples):
            split = (
                "train" if index < train_count
                else "validation" if index < train_count + validation_count
                else "test"
            )
            tensor_path = export_binary_tensor(Path(image_path), processed_dir, image_size)
            records.append({
                "path": image_path,
                "mask_path": mask_path,
                "label": label,
                "split": split,
                "tensor_path": tensor_path,
            })

    write_jsonl(resolve(root, config["manifest"]), records)
    return records, counts


def prepare_animal(root: Path, config: dict, image_size: int = 224) -> tuple[list[dict], Counter]:
    raw_dir = resolve(root, config["raw_dir"])
    if not raw_dir.is_dir() and "raw_dir_fallback" in config:
        fallback = resolve(root, config["raw_dir_fallback"])
        if fallback.is_dir():
            raw_dir = fallback

    range_size = int(config["range_size"])
    class_names = config["class_names"]
    processed_dir = resolve(root, "data/processed/animal")
    samples = []

    # Collect images either directly or recursively, deduplicating by stem name
    candidate_paths = (
        list(raw_dir.iterdir()) if any(f.is_file() for f in raw_dir.iterdir())
        else list(raw_dir.rglob("*"))
    )
    seen_stems = set()

    for image_path in sorted(candidate_paths):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if image_path.stem in seen_stems:
            continue
        # Extract digits from stem (e.g., data12 -> 12, 12 -> 12)
        digits = re.findall(r"\d+", image_path.stem)
        if not digits:
            continue
        numeric_suffix = digits[-1]
        class_index = int(numeric_suffix) // range_size
        if class_index >= len(class_names):
            continue
        seen_stems.add(image_path.stem)
        samples.append((str(image_path.resolve()), class_names[class_index]))

    if not samples:
        raise FileNotFoundError(f"No animal images found in {raw_dir}")

    split_config = config["split"]
    rng = random.Random(int(split_config["seed"]))
    by_class = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)

    records = []
    counts = Counter()
    for label, class_samples in by_class.items():
        rng.shuffle(class_samples)
        train_count = int(len(class_samples) * float(split_config["train"]))
        validation_count = int(len(class_samples) * float(split_config["validation"]))
        for index, (image_path, _) in enumerate(class_samples):
            split = (
                "train" if index < train_count
                else "validation" if index < train_count + validation_count
                else "test"
            )
            tensor_path = export_binary_tensor(Path(image_path), processed_dir, image_size)
            records.append({
                "path": image_path,
                "label": label,
                "split": split,
                "tensor_path": tensor_path,
            })
            counts[f"{split}/{label}"] += 1

    write_jsonl(resolve(root, config["manifest"]), records)
    return records, counts


def prepare_mnist(root: Path, config: dict, image_size: int = 28) -> tuple[list[dict], Counter]:
    raw_dir = resolve(root, config["raw_dir"])
    processed_dir = resolve(root, "data/processed/mnist")
    samples = []
    counts = Counter()

    for image_path in sorted(raw_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label, separator, _ = image_path.stem.partition("_")
        if not separator or not label.isdigit():
            continue
        if label not in config["class_names"]:
            continue
        samples.append((str(image_path.resolve()), label))

    if not samples:
        raise FileNotFoundError(f"No MNIST images found in {raw_dir}")

    split_config = config["split"]
    rng = random.Random(int(split_config["seed"]))
    by_class: dict[str, list[tuple[str, str]]] = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)

    records = []
    for label, class_samples in by_class.items():
        rng.shuffle(class_samples)
        train_count = int(len(class_samples) * float(split_config["train"]))
        validation_count = int(len(class_samples) * float(split_config["validation"]))
        for index, (image_path, _) in enumerate(class_samples):
            split = (
                "train" if index < train_count
                else "validation" if index < train_count + validation_count
                else "test"
            )
            tensor_path = export_binary_tensor(Path(image_path), processed_dir, image_size)
            records.append({
                "path": image_path,
                "label": label,
                "split": split,
                "tensor_path": tensor_path,
            })
            counts[f"{split}/{label}"] += 1

    write_jsonl(resolve(root, config["manifest"]), records)
    return records, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--experiment", choices=["simplicity", "aerial", "animal", "mnist", "all"], default="all")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = resolve(root, args.config)
    with config_path.open(encoding="utf-8") as source:
        datasets = yaml.safe_load(source)["datasets"]

    selected = ["simplicity", "aerial", "animal", "mnist"] if args.experiment == "all" else [args.experiment]
    for name in selected:
        prepare = {
            "simplicity": prepare_simplicity,
            "aerial": prepare_aerial,
            "animal": prepare_animal,
            "mnist": prepare_mnist,
        }[name]
        records, counts = prepare(root, datasets[name], image_size=args.image_size)
        print(f"{name}: {len(records)} images prepared with C++ tensors")
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
