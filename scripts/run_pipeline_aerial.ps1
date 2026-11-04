[CmdletBinding()]
param(
    [ValidateRange(1, 1000)]
    [int]$Epochs = 20,

    [ValidateRange(1, 4096)]
    [int]$BatchSize = 32,

    [ValidateRange(32, 4096)]
    [int]$ImageSize = 224,

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

Write-Host "[3/4] Prepare Aerial data" -ForegroundColor Cyan
Invoke-PythonCommand -ArgumentList @("scripts\prepare_datasets.py", "--experiment", "aerial")

Write-Host "[4/4] Train Aerial CNN + sklearn MLP" -ForegroundColor Cyan
Invoke-PythonCommand -ArgumentList @(
    "scripts\train_cnn.py",
    "--experiment", "aerial",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--image-size", "$ImageSize",
    "--embedding-dim", "$EmbeddingDim"
)

Write-Host "[5/5] Analyze Aerial results" -ForegroundColor Cyan
Invoke-PythonCommand -ArgumentList @("scripts\analyze_results.py", "--experiment", "aerial")

Write-Host "Aerial pipeline complete." -ForegroundColor Green
Write-Host "Checkpoints: $projectRoot\outputs\checkpoints\experiments\aerial"
Write-Host "Reports:     $projectRoot\outputs\reports\aerial"
