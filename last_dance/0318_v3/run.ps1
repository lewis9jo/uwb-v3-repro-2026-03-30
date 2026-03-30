param(
    [ValidateSet("train", "figures")]
    [string]$Mode = "train",
    [ValidateSet("all", "lab", "classroom")]
    [string[]]$Tasks = @("all"),
    [int]$Seed = 44,
    [int]$BatchSize = 128,
    [int]$Epochs = 200,
    [double]$Lr = 0.0008,
    [double]$WeightDecay = 0.0002,
    [switch]$RebuildData,
    [switch]$TraceData,
    [switch]$ClassroomBaseline,
    [switch]$SkipFigure
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$argsList = @(
    "--mode", $Mode,
    "--tasks"
)
$argsList += $Tasks
$argsList += @(
    "--seed", $Seed,
    "--batch-size", $BatchSize,
    "--epochs", $Epochs,
    "--lr", $Lr,
    "--weight-decay", $WeightDecay
)

if ($RebuildData) { $argsList += "--rebuild-data" }
if ($TraceData) { $argsList += "--trace-data" }
if ($ClassroomBaseline) { $argsList += "--classroom-baseline" }
if ($SkipFigure) { $argsList += "--skip-figure" }

python v3_repro_all_in_one.py @argsList

if ($LASTEXITCODE -eq 0) {
    Write-Host "Done. output root: last_dance\\0318_v3\\outputs" -ForegroundColor Green
} else {
    Write-Host "Failed. check logs." -ForegroundColor Red
    exit $LASTEXITCODE
}
