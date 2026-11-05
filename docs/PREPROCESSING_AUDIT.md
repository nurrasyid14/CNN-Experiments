# Preprocessing Repository Audit

## Scope and baseline

Audit date: 2026-11-06.

The repository has a C++ OpenCV ingestion executable and a Python/PyTorch
dataset loader. CUDA Toolkit 12.6 is installed and an NVIDIA GeForce GTX 1050
Ti (compute capability 6.1) is present, but the repository does not currently
enable CUDA compilation or call any CUDA/OpenCV-CUDA API.

## Image loading

| Location | Operation | Backend |
|---|---|---|
| `src/preprocessing/ingestion/image_reader.cpp` | Reads bytes and decodes with `cv::imdecode(..., cv::IMREAD_UNCHANGED)` | CPU/OpenCV |
| `src/preprocessing/python/dataloader.py` | Opens raw images with `PIL.Image.open` when no C++ tensor exists | CPU/PIL |

The C++ reader avoids `cv::imread` path issues on Windows while preserving
corrupt-image detection and format probing.

## Transformations

The C++ implementation resizes with `INTER_AREA`, converts one-, three-, and
four-channel images to RGB, scales UINT8 pixels to `[0, 1]`, normalizes with
ImageNet mean/std values, and writes CHW FLOAT32 output.

The Python loader has a second implementation of these transformations:

- training augmentation with random horizontal flip and random rotation;
- resize;
- PIL-to-tensor conversion;
- ImageNet normalization.

The Python path also bypasses C++ preprocessing when a manifest record does not
contain `tensor_path`.

## Tensor contract

Current binary tensors use:

```text
Magic: 8 bytes "CNNTNS1\0"
Dimensions: 3 little-endian UINT32 values (C, H, W)
Payload: CHANNELS * HEIGHT * WIDTH little-endian FLOAT32 values
Memory order: contiguous CHW
Channel order: RGB
Default shape: 3 x 224 x 224
```

The Python loader reshapes `values` as `CHW`. Its fallback torchvision path
produces CHW float tensors after PIL conversion. No batch tensors are stored;
`torch.utils.data.DataLoader` constructs batches.

## Export mechanism

Only the C++ binary `.bin` export and its Python import implementation exist.
The format has an 8-byte magic and a fixed 12-byte dimensions header. It does
not store dtype, channel order, normalization, layout, or file schema version
as separate metadata. Tensor payload endianness is host-memory byte order,
although the Python reader explicitly requests little-endian values.

## GPU execution

Repository-wide searches found no CUDA language enablement, NVCC sources,
CUDA runtime calls, or `cv::cuda` usage. No CMake CUDA architecture is
configured. `CNN_BUILD_TESTS` only controls CPU C++ tests.

OpenCV vcpkg features do not include `cuda`; the baseline package configuration
reported an ordinary CPU/OpenCV build. Therefore OpenCV CUDA availability must
not be assumed.

## Duplication and replacement plan

| Current behavior | Problem | Replacement |
|---|---|---|
| C++ resize/color/normalization in `image_reader.cpp` | CPU path is coupled to binary export and ingestion | Dedicated CPU transformation and tensor modules |
| Python resize/color/tensor conversion in `dataloader.py` | Duplicates C++ preprocessing and contradicts C++ authority | Consume authoritative `.bin` tensor records only |
| Python augmentation | Training-only policy is not represented in preprocessing configuration | Keep training augmentation optional and explicitly separate from export preprocessing |
| Fixed tensor export header | Metadata is implicit and incomplete | Add a versioned sidecar or manifest fields without changing the `.bin` payload contract unnecessarily |

Existing binary `.bin` data remains supported for compatibility, but the export
implementation must be separated and validated against documented metadata.
Raw-image fallback may remain only as a documented compatibility path until
all dataset records have been exported by C++.

## Build audit

- Root CMake uses C++20 and requires CPU OpenCV core/imgcodecs.
- The selected Visual Studio preset uses a machine-specific vcpkg toolchain
  path in `CMakePresets.json`.
- `vcpkg.json` contains fmt, spdlog, yaml-cpp, nlohmann-json, OpenCV, GTest,
  and Benchmark, but no Eigen and no CUDA package declarations.
- No optional CUDA detection exists.
- `sm_61` is compatible with the target GTX 1050 Ti and CUDA Toolkit 12.6, but
  CMake compatibility must be tested before that architecture is committed.

## Audit findings and implementation priority

1. Preserve the existing ingestion executable, binary format, and tests.
2. Separate image loading, CPU transformations, tensor construction, and export.
3. Add a complete preprocessing configuration including backend and batch
   contract fields.
4. Add native CUDA kernels with CPU fallback; do not depend on OpenCV CUDA.
5. Update Python to validate and consume C++ output without repeating raw-image
   preprocessing.
6. Add correctness tests, CUDA/CPU consistency tests where CUDA is available,
   and benchmarks that report evidence rather than assumptions.

## CUDA toolchain verification (2026-11-06)

NVCC 12.6 exists on disk, and the target GPU reports compute capability 6.1.
CMake could not detect CUDA automatically with the Visual Studio 2022 Build
Tools generator. Passing the explicit NVCC path also failed with
`No CUDA toolset found`, which indicates that the CUDA Visual Studio
integration required by CMake is missing from this installation (rather than
that the GTX 1050 Ti cannot run CUDA 12.6).

The Visual Studio MSBuild integration is absent, so Visual Studio generator
CUDA builds are not available until the official CUDA BuildCustomizations are
registered by an elevated installer. Native CUDA compilation was repaired using
the initialized MSVC/NMake toolchain and completed successfully.
