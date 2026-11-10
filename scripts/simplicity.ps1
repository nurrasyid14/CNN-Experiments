[CmdletBinding()]
param(
    [ValidateRange(1, 1000)]
    [int]$Epochs = 22,

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

$scriptPath = $MyInvocation.MyCommand.Path
$scriptDir = Split-Path -Parent $scriptPath
$projectRoot = Split-Path -Parent $scriptDir
$rootRunner = Join-Path $projectRoot "simplicity.ps1"

& $rootRunner @PSBoundParameters

