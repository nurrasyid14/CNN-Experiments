from __future__ import annotations

import argparse
import csv
import json
import random
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


def prepare_simplicity(root: Path, config: dict) -> tuple[list[dict], Counter]:
    raw_dir = resolve(root, config["raw_dir"])
    divisor = int(config["label_divisor"])
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
            records.append({"path": str(image_path.resolve()), "label": label, "split": split})
            counts[f"{split}/class_{label}"] += 1

    write_jsonl(resolve(root, config["manifest"]), records)
    return records, counts


def prepare_aerial(root: Path, config: dict) -> tuple[list[dict], Counter]:
    raw_dir = resolve(root, config["raw_dir"])
    images_dir = raw_dir / config["images_dir"]
    masks_dir = raw_dir / config["masks_dir"]
    class_names, colors = load_classes(resolve(root, config["class_csv"]))
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
            records.append({
                "path": image_path,
                "mask_path": mask_path,
                "label": label,
                "split": split,
            })

    write_jsonl(resolve(root, config["manifest"]), records)
    return records, counts


def prepare_animal(root: Path, config: dict) -> tuple[list[dict], Counter]:
    raw_dir = resolve(root, config["raw_dir"])
    range_size = int(config["range_size"])
    class_names = config["class_names"]
    samples = []

    for image_path in sorted(raw_dir.iterdir(), key=lambda path: int(path.stem[4:])):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        numeric_suffix = image_path.stem[4:]
        if not numeric_suffix.isdigit():
            raise ValueError(f"Expected a numeric animal image name: {image_path}")
        class_index = int(numeric_suffix) // range_size
        if class_index >= len(class_names):
            raise ValueError(f"No class configured for {image_path.name}")
        samples.append((str(image_path.resolve()), class_names[class_index]))

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
            records.append({"path": image_path, "label": label, "split": split})
            counts[f"{split}/{label}"] += 1

    write_jsonl(resolve(root, config["manifest"]), records)
    return records, counts


def prepare_mnist(root: Path, config: dict) -> tuple[list[dict], Counter]:
    raw_dir = resolve(root, config["raw_dir"])
    samples = []
    counts = Counter()

    for image_path in sorted(raw_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label, separator, _ = image_path.stem.partition("_")
        if not separator or not label.isdigit():
            raise ValueError(f"Expected a labeled MNIST image name: {image_path.name}")
        if label not in config["class_names"]:
            raise ValueError(f"No class configured for {image_path.name}")
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
            records.append({"path": image_path, "label": label, "split": split})
            counts[f"{split}/{label}"] += 1

    write_jsonl(resolve(root, config["manifest"]), records)
    return records, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--experiment", choices=["simplicity", "aerial", "animal", "mnist", "all"], default="all")
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
        records, counts = prepare(root, datasets[name])
        print(f"{name}: {len(records)} images")
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
