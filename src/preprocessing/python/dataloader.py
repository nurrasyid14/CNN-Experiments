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
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.allow_raw_fallback = allow_raw_fallback

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
            return tensor.astype(np.float32, copy=False), int(record["label_index"])
        raise ValueError(
            f"Record has no C++ tensor_path: {record.get('path', '<unknown>')}"
        )