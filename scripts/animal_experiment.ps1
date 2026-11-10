[CmdletBinding()]
param(
    [ValidateRange(1, 1000)]
    [int]$Epochs = 15,

    [ValidateRange(1, 4096)]
    [int]$BatchSize = 8,

    [ValidateRange(32, 4096)]
    [int]$ImageSize = 224,

    [ValidateRange(1, 4096)]
    [int]$EmbeddingDim = 1280,

    [string]$Architecture = "repvgg-a0",

    [string]$PythonExecutable = "",

    [switch]$SkipNativeBuild
)

# -----------------------------------------------------------------
# Resolve the location of the project‑root script (E:\CNN-Experiments\animal_experiment.ps1)
# -----------------------------------------------------------------
# Full path of this wrapper script (…\scripts\animal_experiment.ps1)
$scriptPath = $MyInvocation.MyCommand.Path

# Directory that contains this wrapper (the “scripts” folder)
$scriptDir = Split-Path -Parent $scriptPath

# Project root (one level up from the scripts folder)
$projectRoot = Split-Path -Parent $scriptDir

# Full path to the root‑level script we actually want to run
$rootRunner = Join-Path $projectRoot "animal_experiment.ps1"

# -----------------------------------------------------------------
# Forward every bound parameter to the root script
# -----------------------------------------------------------------
# The splatting operator (@) expands the hashtable $PSBoundParameters
# into the individual arguments expected by the target script.
& $rootRunner @PSBoundParameters