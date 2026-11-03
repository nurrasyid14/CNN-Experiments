[CmdletBinding()]
param(
    [ValidateRange(1, 1000)]
    [int]$Epochs = 12,

    [ValidateRange(1, 4096)]
    [int]$BatchSize = 64,

    [ValidateRange(28, 4096)]
    [int]$ImageSize = 28,

    [ValidateRange(1, 4096)]
    [int]$EmbeddingDim = 512,

    [string]$PythonExecutable = "py",

    [switch]$SkipNativeBuild
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

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

    if ([IO.Path]::GetFileNameWithoutExtension($PythonExecutable).ToLowerInvariant() -eq "py") {
        Invoke-ProjectCommand -FilePath $PythonExecutable -ArgumentList (@("-3") + $ArgumentList)
    }
    else {
        Invoke-ProjectCommand -FilePath $PythonExecutable -ArgumentList $ArgumentList
    }
}

if (-not $SkipNativeBuild) {
    Write-Host "[1/4] Configure CMake" -ForegroundColor Cyan
    Invoke-ProjectCommand -FilePath "cmake" -ArgumentList @("--preset", "windows-release")

    Write-Host "[2/4] Build CMake targets" -ForegroundColor Cyan
    Invoke-ProjectCommand -FilePath "cmake" -ArgumentList @("--build", "build", "--preset", "windows-release")
}

Write-Host "[3/4] Prepare MNIST data" -ForegroundColor Cyan
$tensorManifest = Join-Path $projectRoot "data\manifests\mnist.jsonl"
$tensorDirectory = Join-Path $projectRoot "data\processed\mnist"
Invoke-ProjectCommand -FilePath (Join-Path $projectRoot "build\ingestion\Release\cnn_ingest.exe") -ArgumentList @(
    "--root", (Join-Path $projectRoot "data\raw\mnist\train"),
    "--out", $tensorManifest,
    "--label-mode", "filename-prefix",
    "--export-tensors", $tensorDirectory,
    "--image-size", "$ImageSize"
)

Write-Host "[4/4] Train MNIST CNN + sklearn MLP" -ForegroundColor Cyan
Invoke-PythonCommand -ArgumentList @(
    "scripts\train_cnn.py",
    "--experiment", "mnist",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--image-size", "$ImageSize",
    "--embedding-dim", "$EmbeddingDim"
)

Write-Host "[5/5] Analyze MNIST results" -ForegroundColor Cyan
Invoke-PythonCommand -ArgumentList @("scripts\analyze_results.py", "--experiment", "mnist")

Write-Host "MNIST pipeline complete." -ForegroundColor Green
Write-Host "Checkpoints: $projectRoot\checkpoints\experiments\mnist"
Write-Host "Reports:     $projectRoot\reports\mnist"
