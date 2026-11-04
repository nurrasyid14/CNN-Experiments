# CNN Classifier

A reproducible PyTorch CNN feature-learning project for two image datasets:

- **SIMPLIcity:** 1,000 images in predefined train, validation, and test splits.
- **Aerial:** 10 sampled RGB images with semantic masks mapped to the class definitions in
  `class_dict_seg.csv`.

The implementation uses a PyTorch CNN to learn image embeddings, then uses a regular
scikit-learn `MLPClassifier` as the final classifier. This structure separates CNN feature
learning from the sklearn terminal MLP.

## Pipeline

```text
raw images
  -> dataset manifest
  -> PyTorch CNN feature extractor
  -> frozen embeddings
  -> sklearn MLPClassifier
  -> predictions and metrics
```

## Environment

Run C++ commands from an MSVC x64 developer prompt:

```bat
cmake --preset windows-release
cmake --build build --preset windows-release
ctest --test-dir build -C Release --output-on-failure
```

Set up Python with CUDA-enabled PyTorch:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .[dev]
python scripts\check_env.py
```

## Prepare datasets

```bat
python scripts\prepare_datasets.py
python scripts\prepare_datasets.py --experiment simplicity
python scripts\prepare_datasets.py --experiment aerial
```

Dataset settings are in [configs/datasets.yaml](configs/datasets.yaml). Manifests are written
to [data/manifests/](data/manifests/) in JSON Lines format.

## Train and classify

```bat
python scripts\train_cnn.py --experiment simplicity --epochs 10 --batch-size 32
python scripts\train_cnn.py --experiment aerial --epochs 10 --batch-size 32
```

Optional arguments:

- `--image-size`: input image side length, default `224`.
- `--embedding-dim`: CNN feature vector length, default `512`.
- `--batch-size`: PyTorch training and feature-extraction batch size.
- `--epochs`: number of CNN training epochs.

Results are written to `outputs/reports/<experiment>/metrics.json`; frozen CNN weights and class names
are written to `outputs/reports/<experiment>/cnn_features.pt`.

## Project structure

```text
CNN-Classifier/
├── CMakeLists.txt
├── CMakePresets.json
├── vcpkg.json
├── src/
│   ├── convolution/           # CNN models and architecture definitions
│   ├── learning/              # Training evaluation and reporting
│   ├── preprocessing/         # C++ and Python ingestion/data loading
│   ├── pipeline/              # Training pipeline entry points
│   └── utils/                 # Metrics counters and visualisation helpers
├── scripts/
│   ├── prepare_datasets.py    # Create both dataset manifests
│   ├── train_cnn.py           # CNN training + sklearn MLP classification
│   └── check_env.py           # CUDA environment check
├── configs/
│   ├── datasets.yaml
│   ├── model.yaml
│   └── training.yaml
├── data/raw/
├── data/manifests/
└── outputs/
    ├── checkpoints/
    └── reports/
```

## Model behavior

`CNNFeatureExtractor` applies convolution, batch normalization, ReLU, pooling, and adaptive
average pooling. It uses a temporary PyTorch linear head during CNN training. After training,
`embed()` removes that head and exports fixed feature vectors.

The sklearn MLP then:

1. Fits on training CNN embeddings.
2. Predicts test embeddings.
3. Reports accuracy and a complete classification report.

SIMPLIcity preserves its predefined splits. Aerial uses the configured stratified split when
preparing its manifest.

## Reproducibility

Training uses a configured random seed. CUDA and CPU availability are reported in the output
metrics. Dataset manifests preserve image paths and labels; they should be regenerated after
changing the raw dataset layout.

## Verification

```bat
python -m compileall model training scripts
python scripts\train_cnn.py --experiment aerial --epochs 1
ctest --test-dir build -C Release --output-on-failure
```

## Windows pipeline scripts

Run the complete native build, dataset preparation, and both training experiments:

```bat
scripts\run_pipeline.bat
```

Run one experiment from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_pipeline_simplicity.ps1 -Epochs 20 -BatchSize 32
powershell -ExecutionPolicy Bypass -File .\scripts\run_pipeline_aerial.ps1 -Epochs 20 -BatchSize 32
```

If the native build is already current, add `-SkipNativeBuild` to run only dataset preparation
and training:

```powershell
.\scripts\run_pipeline_aerial.ps1 -Epochs 10 -SkipNativeBuild
```

Training pipelines automatically produce ROC-AUC, PR-AUC, confusion matrices, training and
gradient-descent curves, and `analysis.json` under each experiment's report directory.

Override defaults with environment variables:

```bat
set CNN_EPOCHS=20
set CNN_BATCH_SIZE=32
scripts\run_pipeline.bat
```

The legacy C++ manifest command can be run with:

```bat
scripts\ingest_simplicity.bat
```

## Checkpoints

Each experiment stores PyTorch checkpoints in `outputs/checkpoints/experiments/<experiment>/`:

- `best_model.pt`: best CNN according to validation accuracy.
- `latest_model.pt`: final CNN epoch and optimizer state.
- `sklearn_mlp.joblib`: fitted final sklearn MLP.
- `cnn_features.pt`: CNN weights used to generate the embeddings.
