import numpy as np

from preprocessing.python.dataloader import ManifestDataset


def _write_tensor(path, tensor: np.ndarray) -> None:
    with path.open("wb") as target:
        target.write(b"CNNTNS1\0")
        target.write(np.array(tensor.shape, dtype="<u4").tobytes())
        target.write(tensor.astype("<f4", copy=False).tobytes())


def test_augmentation_is_disabled_when_not_requested(tmp_path) -> None:
    tensor = np.arange(3 * 32 * 32, dtype=np.float32).reshape(3, 32, 32)
    tensor_path = tmp_path / "sample.bin"
    _write_tensor(tensor_path, tensor)
    dataset = ManifestDataset(
        [{"tensor_path": str(tensor_path), "label_index": 0}],
        image_size=(32, 32),
        training=True,
        augment=False,
    )

    image, label = dataset[0]

    assert label == 0
    np.testing.assert_array_equal(image, tensor)
