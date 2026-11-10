[CmdletBinding()]
param(
    [ValidateRange(1, 1000)]
    [int]$Epochs = 10,

    [ValidateRange(1, 4096)]
    [int]$BatchSize = 32,

    [ValidateRange(32, 4096)]
    [int]$ImageSize = 224,

    [ValidateRange(1, 4096)]
    [int]$EmbeddingDim = 1280,

    [string]$Architecture = "repvgg-a0",

    [string]$PythonExecutable = "",

    [switch]$SkipNativeBuild
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path (Join-Path $projectRoot "configs"))) {
    $projectRoot = Split-Path -Parent $projectRoot
}
Set-Location $projectRoot

function Resolve-PythonPath {
    param([string]$ExplicitPath)
    if ($ExplicitPath -and (Test-Path $ExplicitPath)) {
        return $ExplicitPath
    }
    $candidates = @(
        "C:\Users\Muhamad Nur Rasyid\AppData\Local\Programs\Python\Python312\python.exe",
        (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
        (Get-Command py -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            try {
                $test = & $candidate -c "import torch; print(torch.__version__)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $test) {
                    return $candidate
                }
            } catch {}
        }
    }
    throw "A compatible Python with PyTorch was not found."
}

$pyPath = Resolve-PythonPath -ExplicitPath $PythonExecutable

function Invoke-ProjectCommand {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,
        [string[]]$ArgumentList = @()
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

function Invoke-PythonCommand {
    param([string[]]$ArgumentList)
    Invoke-ProjectCommand -FilePath $pyPath -ArgumentList $ArgumentList
}

# 1. Native C++ build check
$ingestExe = Join-Path $projectRoot "build\src\preprocessing\Release\cnn_ingest.exe"
if (-not $SkipNativeBuild -and -not (Test-Path $ingestExe)) {
    Write-Host "[1/4] Build C++ preprocessing tools" -ForegroundColor Cyan
    Invoke-ProjectCommand -FilePath "cmake" -ArgumentList @("--preset", "windows-release")
    Invoke-ProjectCommand -FilePath "cmake" -ArgumentList @("--build", "build", "--preset", "windows-release", "--target", "cnn_ingest")
}

# 2. C++ Ingestion & Tensor Generation for SIMPLIcity
Write-Host "[2/4] Preprocessing SIMPLIcity data via C++ (cnn_ingest.exe -> CNNTNS1 tensors)" -ForegroundColor Cyan
$manifestPath = Join-Path $projectRoot "data\manifests\simplicity.jsonl"
$rawDir = Join-Path $projectRoot "data\raw\SIMPLIcity Dataset"
$processedDir = Join-Path $projectRoot "data\processed\simplicity"

if (Test-Path $ingestExe) {
    Invoke-ProjectCommand -FilePath $ingestExe -ArgumentList @(
        "--root", $rawDir,
        "--out", $manifestPath,
        "--label-mode", "numeric-stem",
        "--label-divisor", "100",
        "--export-tensors", $processedDir,
        "--image-size", "$ImageSize"
    )
} else {
    Invoke-PythonCommand -ArgumentList @(
        "scripts\prepare_datasets.py",
        "--experiment", "simplicity",
        "--image-size", "$ImageSize"
    )
}

# 3. Generate experiment ID: <YYYYMMDD_HHMMSS>_<dataset>
$runTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runId = "${runTimestamp}_simplicity"
$experimentDir = Join-Path $projectRoot "experiments\results\$runId"

Write-Host "[3/4] Running experiment with RepVGG-A0: $runId" -ForegroundColor Cyan
Invoke-PythonCommand -ArgumentList @(
    "scripts\train_cnn.py",
    "--experiment", "simplicity",
    "--architecture", $Architecture,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--image-size", "$ImageSize",
    "--embedding-dim", "$EmbeddingDim",
    "--run-id", $runId
)

# 4. Verification and artifact channeling
Write-Host "[4/4] Channeling outputs and verifying artifacts..." -ForegroundColor Cyan

$bestModelPt = Join-Path $experimentDir "best_model.pt"
$featsPt     = Join-Path $experimentDir "feats.pt"
$metricsJson = Join-Path $experimentDir "metrics.json"
$metricsLog  = Join-Path $experimentDir "metrics.log"
$figuresDir  = Join-Path $experimentDir "figures"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " EXPERIMENT COMPLETED SUCCESSFULLY: $runId" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Architecture       : $Architecture" -ForegroundColor Yellow
Write-Host " Experiment Directory: $experimentDir"
Write-Host " Best Params (.pt)  : $bestModelPt"
Write-Host " Features (.pt)     : $featsPt"
Write-Host " Decoded Metrics    : $metricsJson"
Write-Host " Metrics Log        : $metricsLog"
Write-Host " Generated Figures  : $figuresDir"
if (Test-Path $figuresDir) {
    Get-ChildItem $figuresDir -Filter "*.png" | ForEach-Object {
        Write-Host "   - $($_.Name)" -ForegroundColor DarkGray
    }
}
Write-Host "============================================================" -ForegroundColor Green

