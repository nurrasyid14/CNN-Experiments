# ENGINEERING DIRECTIVE
## Refactoring CNN-Experiments: GPU-Accelerated Preprocessing Architecture

### 1. Primary Objective

Refactor the existing preprocessing subsystem to establish a unified C++/CUDA preprocessing pipeline before proceeding with advanced CNN or ConvLSTM architecture development.

The target architecture must establish:

- C++ as the primary preprocessing environment.
- OpenCV as the primary image-processing library.
- CUDA C++ as the GPU acceleration layer.
- Eigen as an optional numerical computation library.
- Python/PyTorch as the downstream training and learning environment.

Do not prematurely implement CNN, ConvLSTM, or advanced neural-network architectures.

Prioritize preprocessing correctness, reproducibility, interoperability, and performance measurement.

---

### 2. Phase I — Repository Audit

Inspect the existing repository before modifying anything.

Relevant locations include:

- `CMakeLists.txt`
- `configs/preprocessing.yaml`
- `src/preprocessing/ingestion/main.cpp`
- `src/preprocessing/ingestion/image_reader.cpp`
- `src/preprocessing/python/dataloader.py`
- `src/pipeline/runners/run_training.py`
- Existing headers, utility modules, tests, and build configurations.

Identify:

1. All image-loading operations.
2. All image transformations.
3. All normalization operations.
4. All tensor construction operations.
5. Existing CPU/GPU execution paths.
6. Existing data-export mechanisms.
7. Duplicated preprocessing logic between C++ and Python.
8. Existing CUDA and OpenCV CUDA availability.
9. Current tensor dimensions, channel ordering, and data types.

Produce an audit report before modifying the implementation.

Do not delete existing functionality without documenting its replacement.

### 3. Phase II — Establish the Preprocessing Architecture

Refactor the C++ source tree into the following structure:

```text
src/
├── preprocessing/
│   ├── ingestion/
│   │   ├── image_reader.cpp
│   │   ├── dataset_loader.cpp
│   │   └── image_validator.cpp
│   │
│   ├── transformation/
│   │   ├── resize.cpp
│   │   ├── color_conversion.cpp
│   │   ├── augmentation.cpp
│   │   └── normalization.cpp
│   │
│   ├── tensor/
│   │   ├── tensor_builder.cpp
│   │   ├── tensor_layout.cpp
│   │   └── batch_builder.cpp
│   │
│   ├── cuda/
│   │   ├── cuda_context.cpp
│   │   ├── cuda_memory.cu
│   │   ├── image_transform.cu
│   │   ├── normalization.cu
│   │   └── kernels.cuh
│   │
│   ├── export/
│   │   ├── numpy_export.cpp
│   │   └── binary_export.cpp
│   │
│   └── include/
│       ├── image_reader.hpp
│       ├── tensor_builder.hpp
│       ├── preprocessing.hpp
│       └── cuda_backend.hpp
│
├── pipeline/
│   ├── preprocessing_pipeline.cpp
│   ├── dataset_pipeline.cpp
│   └── execution_manager.cpp
│
└── ...
```

Adjust the structure where existing project conventions justify it, but maintain clear separation between CPU preprocessing, CUDA execution, tensor construction, and pipeline orchestration.

### 4. Phase III — CPU Preprocessing

Implement and validate the CPU preprocessing backend first.

Use OpenCV for:

- Image decoding.
- Image dimension validation.
- RGB/BGR conversion.
- Resizing.
- Cropping.
- Image augmentation.
- Image transformations.

Use Eigen only where general matrix operations or numerical computations justify its inclusion.

Do not introduce redundant Eigen operations where OpenCV already provides an appropriate implementation.

All transformations must be configurable through:

`configs/preprocessing.yaml`

Required configuration parameters:

- Input image dimensions.
- Output image dimensions.
- Color format.
- Interpolation method.
- Normalization method.
- Mean and standard deviation.
- Batch size.
- Tensor dtype.
- Tensor layout.
- Backend selection.

Maintain CPU execution as a reliable fallback.

### 5. Phase IV — CUDA Preprocessing Backend

Implement a CUDA backend using `.cu` source files.

CUDA operations should initially cover:

1. GPU memory allocation and release.
2. Host-to-device data transfer.
3. Image normalization.
4. Element-wise transformations.
5. Batch-oriented image operations where beneficial.

Investigate OpenCV CUDA availability separately.

Do not assume that the installed OpenCV vcpkg package includes CUDA modules.

If OpenCV CUDA is unavailable, implement necessary operations through native CUDA kernels rather than making the entire project dependent on a custom OpenCV build.

Requirements:

- Use CUDA error checking.
- Avoid unnecessary memory transfers.
- Support batch processing.
- Handle GPU memory limits.
- Use appropriate CUDA synchronization.
- Avoid implicit assumptions about image dimensions.
- Support CPU fallback when CUDA is unavailable.

Do not implement a complete custom CNN convolution engine in this phase.

### 6. Phase V — Tensor Construction

Establish a strict tensor contract between C++ and Python.

Default input:

- RGB image.
- 224 × 224 pixels.
- Three channels.

Expected PyTorch tensor:

`B × C × H × W`

Example:

`32 × 3 × 224 × 224`

Required operations:

- HWC to CHW conversion.
- UINT8 to FLOAT32 conversion.
- Pixel scaling.
- Configurable normalization.
- Batch construction.
- Tensor shape validation.

For future ConvLSTM support, design the interface so that temporal sequences can later be represented as:

`B × T × C × H × W`

Do not fabricate temporal dimensions for ordinary image classification.

### 7. Phase VI — C++ / Python Interoperability

Make C++ the authoritative preprocessing implementation.

Python must not independently repeat image transformations already completed by C++.

Implement a documented export/import interface.

Preferred initial formats:

- NumPy `.npy` for straightforward interoperability.
- Binary `.bin` with explicit metadata for performance experiments.

Document:

- Shape.
- Dtype.
- Memory order.
- Channel order.
- Normalization parameters.
- Dataset sample ordering.
- Label mapping.

Modify `src/preprocessing/python/dataloader.py` to consume the processed output instead of repeating the original image preprocessing.

Preserve the ability to load labels and construct PyTorch datasets.

Do not introduce unnecessary serialization complexity.

### 8. Phase VII — Testing and Benchmarking

Use GoogleTest and Google Benchmark.

Required tests:

- Image decoding correctness.
- Image dimensions.
- Color channel correctness.
- Normalization correctness.
- Tensor layout correctness.
- Batch construction.
- CPU/GPU numerical consistency.
- Export/import consistency.
- Invalid image handling.
- CUDA unavailable fallback.

Benchmark:

1. CPU image preprocessing.
2. CUDA image preprocessing.
3. CPU normalization.
4. CUDA normalization.
5. Batch processing throughput.
6. Host-to-device transfer overhead.
7. End-to-end preprocessing throughput.

Report:

- Processing time.
- Images per second.
- Memory consumption.
- GPU utilization where measurable.
- CPU/GPU numerical deviation.

Do not claim GPU acceleration is beneficial without benchmark evidence.

### 9. Phase VIII — Build System

Refactor CMake to support:

- C++17 or the project's existing compatible standard.
- OpenCV.
- Eigen3.
- fmt.
- spdlog.
- yaml-cpp.
- nlohmann-json.
- GoogleTest.
- Google Benchmark.
- Optional CUDA compilation.

Use `check_language(CUDA)` or equivalent capability detection.

Do not make CUDA an unconditional requirement for CPU builds.

Ensure that `.cu` files are compiled using NVCC and that CUDA architectures are configured explicitly.

Target hardware:

- NVIDIA GeForce GTX 1050 Ti.
- Compute Capability 6.1.
- CUDA architecture `sm_61`.

Verify actual compatibility with the installed CUDA Toolkit and compiler before committing the build configuration.

Do not silently remove CUDA support if the current toolchain is incompatible. Report the precise incompatibility and preserve a functioning CPU build.

### 10. Phase IX — Completion Criteria

The refactoring is complete only when:

- The C++ CPU preprocessing pipeline works independently.
- CUDA preprocessing can be enabled or disabled.
- The CPU fallback works.
- The tensor contract is documented.
- Python can consume C++-generated data.
- No duplicated preprocessing remains without justification.
- Unit tests pass.
- CPU/GPU benchmarks are available.
- CMake builds the project reproducibly.

### 11. Deliverables

Produce:

1. Repository audit report.
2. Updated directory tree.
3. Refactored C++ source files.
4. CUDA kernel implementations.
5. Updated CMakeLists.txt.
6. Updated preprocessing.yaml.
7. Updated Python dataloader.
8. Unit tests.
9. Benchmark results.
10. Technical documentation explaining the data pipeline.

### Execution Rules

Work incrementally.

Do not replace the entire project in one operation.

Inspect existing files before editing them.

Preserve working implementations.

Build and test after each major phase.

Report incomplete features explicitly.

Do not proceed to advanced CNN/ConvLSTM architecture development until all preprocessing completion criteria have been satisfied.