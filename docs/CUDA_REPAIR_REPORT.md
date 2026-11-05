# CUDA Toolchain Repair Report

Date: 2026-11-06

## Root cause

NVCC 12.6.20 and MSVC 19.44.35229 are both available and compile together in
an initialized `VsDevCmd` environment. The Visual Studio generator failed with
`No CUDA toolset found` because this Build Tools installation does not contain
the CUDA 12.6 MSBuild customizations:

- `CUDA 12.6.props`
- `CUDA 12.6.targets`
- `CUDA 12.6.xml`
- `Nvda.Build.CudaTasks.v12.6.dll`

Official copies are present in the CUDA Toolkit installation, but registration
under Build Tools requires elevation. The project avoids generating or
overwriting those official files.

## Repair

The build now discovers CUDA and NVCC in the initialized NMake/MSVC
environment, forces architecture 61, and restricts MSVC-only compiler options
to C++ sources. CUDA language support is optional.

Native source compiled to:

```text
cnn_preprocessing_cuda.lib
```

## Validation

| Configuration | Result |
|---|---|
| CUDA NMake build | Passed, CUDA library and executable linked |
| CUDA runtime normalization test | Passed on GTX 1050 Ti |
| CPU NMake build | Passed |
| CPU CTest suite | Passed |
| CUDA/CPU numeric consistency | Passed within 1e-5 |

## Benchmark

The current benchmark performs 100 NCHW float32 normalization iterations for
one 224 x 224 RGB image. CUDA timing includes allocations, transfers, kernel,
and device synchronization:

```text
CPU: 267.395 ms total, 373.978 images/second
CUDA end-to-end: 88.249 ms total, 1133.16 images/second
```

This validates normalization transfer overhead and kernel execution, not
end-to-end decoding, resize, color conversion, and export throughput.

## Remaining limitation

Visual Studio `.sln`/MSBuild builds still require the official CUDA 12.6
integration files to be registered in Build Tools. The supported current path
is NMake in the initialized `VsDevCmd` environment.
