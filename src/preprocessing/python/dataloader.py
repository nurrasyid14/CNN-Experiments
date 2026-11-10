import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    return records


def encode_labels(records: list[dict]) -> tuple[list[dict], list[str], dict[str, int]]:
    values = sorted({record["label"] for record in records}, key=lambda value: str(value))
    names = [str(value) for value in values]
    mapping = {name: index for index, name in enumerate(names)}
    encoded = [
        {**record, "label_index": mapping[str(record["label"])]}
        for record in records
    ]
    return encoded, names, mapping


class ManifestDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        image_size: tuple[int, int] = (224, 224),
        training: bool = False,
        allow_raw_fallback: bool = False,
        augment: bool = True,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.allow_raw_fallback = allow_raw_fallback
        self.training = training
        self.augment = augment
        self.records = records
        self.image_size = image_size
        self.allow_raw_fallback = allow_raw_fallback
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[object, int]:
        record = self.records[index]
        if "tensor_path" in record:
            tensor_path = Path(record["tensor_path"])
            with tensor_path.open("rb") as source:
                magic = source.read(8)
                dimensions = np.frombuffer(source.read(12), dtype="<u4")
                values = np.frombuffer(source.read(), dtype="<f4")
            if magic != b"CNNTNS1\0":
                raise ValueError(f"Invalid tensor file: {tensor_path}")
            channels, height, width = (int(value) for value in dimensions)
            if (channels, height, width) != (
                3, self.image_size[0], self.image_size[1]
            ):
                raise ValueError(
                    f"Unexpected tensor shape {(channels, height, width)} "
                    f"for {tensor_path}; expected 3 x {self.image_size[0]} "
                    f"x {self.image_size[1]}"
                )
            tensor = values.reshape(channels, height, width).copy()

            if self.training:
                # 1. Random horizontal flip (50% probability)
                if np.random.rand() > 0.5:
                    tensor = np.flip(tensor, axis=2).copy()
                # 2. Random spatial shift (+/- 8 pixels)
                dy = int(np.random.randint(-8, 9))
                dx = int(np.random.randint(-8, 9))
                tensor = np.roll(tensor, shift=(dy, dx), axis=(1, 2)).copy()
                # 3. Subtle contrast / brightness variation (0.9 to 1.1)
                factor = float(np.random.uniform(0.9, 1.1))
                tensor = tensor * factor
                # 4. Random cutout (33% probability): mask a small 24x24 patch
                if np.random.rand() < 0.33:
                    h, w = self.image_size
                    ph, pw = max(16, int(h * 0.12)), max(16, int(w * 0.12))
                    y0 = int(np.random.randint(0, h - ph))
                    x0 = int(np.random.randint(0, w - pw))
                    tensor[:, y0:y0 + ph, x0:x0 + pw] = 0.0

            return tensor.astype(np.float32, copy=False), int(record["label_index"])
        raise ValueError(
            f"Record has no C++ tensor_path: {record.get('path', '<unknown>')}"
        )