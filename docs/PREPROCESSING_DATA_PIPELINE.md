# Preprocessing Data Pipeline

## Authoritative CPU path

1. `cnn_ingest` scans and validates image files.
2. OpenCV decodes each image to a CPU `cv::Mat`.
3. The tensor builder resizes and converts the image to RGB.
4. Pixels are converted to float32 and scaled to `[0, 1]`.
5. ImageNet mean/std normalization is applied.
6. The binary exporter writes one `CHW` float32 tensor per image.

Python training does not resize, color-convert, scale, or normalize these
tensors. It only loads and validates the C++ artifact.

## Binary tensor contract

```text
Offset  Size   Field
0       8       Magic: "CNNTNS1\0"
8       12      Dimensions: C, H, W as three little-endian UINT32 values
20      3*H*W*4 FLOAT32 payload in CHW/RGB planar order
```

The default payload shape is:

```text
3 x 224 x 224 float32
```

Normalization is:

```text
(pixel / 255 - mean[channel]) / std[channel]
```

Mean values are RGB order. The dataset manifest or its tensor record must carry
the same dimensions and label mapping used by `ManifestDataset`.

## CUDA status

Native CUDA normalization sources are present behind
`CNN_ENABLE_CUDA=ON`, targeting `sm_61`. CUDA compilation requires the NVIDIA
Visual Studio integration. Visual Studio generator reports `No CUDA toolset found` because the CUDA
MSBuild customizations are absent. This repository also supports CUDA through
the NMake generator initialized by `VsDevCmd`. Native compilation, linking,
runtime normalization, and end-to-end normalization benchmarking have been
validated on the GTX 1050 Ti. The CPU implementation remains the fallback.

## Build commands

Initialize MSVC/NVCC once:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
```

CUDA build:

```bat
cmake -S . -B build-cuda -G "NMake Makefiles" -DENABLE_CUDA=ON
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
```

CPU-only build:

```bat
cmake -S . -B build-cpu -G "NMake Makefiles" -DENABLE_CUDA=OFF
cmake --build build-cpu
ctest --test-dir build-cpu --output-on-failure
```

## Batches

Individual tensors use `CHW`. `DataLoader` constructs `B x C x H x W`; this
project does not fabricate a temporal `T` dimension for image classification.
Future sequence support should use `B x T x C x H x W` only in sequence-backed
datasets.
