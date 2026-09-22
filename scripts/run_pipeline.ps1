<#
.SYNOPSIS
    Run the full pdbenergy pipeline and log every stage.

.DESCRIPTION
    Chains the CLI steps with sensible defaults and writes per-stage logs into
    logs/.  Long-running physics work is the `ensemble` step; everything after it
    is minutes.

    Stages:
      1. download   fetch the PDB entries from RCSB
      2. inventory  report residue counts / chains / models and flag bad entries
      3. ensemble   generate conformations and label them with AMBER14 + GBn2
      4. dataset    build the protein-level split
      5. train      fit the graph network and the descriptor baseline
      6. evaluate   metrics, per-group breakdowns, figures
      7. ablate     measure how much a leaky frame-level split flatters the model
      8. predict    score a real PDB file and, with -Verify, check against physics

.EXAMPLE
    .\scripts\run_pipeline.ps1
    .\scripts\run_pipeline.ps1 -Preset quick -Ids 1CRN,1L2Y,1VII,5PTI
    .\scripts\run_pipeline.ps1 -SkipEnsemble     # reuse cached ensembles
#>
[CmdletBinding()]
param(
    [string]   $Python   = ".\.venv\Scripts\python.exe",
    [string]   $Preset   = "default",
    [string[]] $Ids      = @(),
    [int]      $Threads  = 0,          # 0 = number of logical processors
    [switch]   $SkipDownload,
    [switch]   $SkipEnsemble,
    [switch]   $SkipTrain,
    [switch]   $Verify,
    [string]   $PredictPdb = "data\raw\1CRN.pdb"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path logs | Out-Null

if ($Threads -le 0) { $Threads = [Environment]::ProcessorCount }
if (-not (Test-Path $Python)) { throw "python not found at '$Python' (create the venv first)" }

$common = @("--preset", $Preset)
if ($Ids.Count -gt 0) { $common += @("--ids") + $Ids }

function Invoke-Stage {
    param([string] $Name, [string[]] $CliArgs)
    $log = "logs\$Name.log"
    Write-Host ""
    Write-Host ("=" * 78)
    Write-Host "== $Name"
    Write-Host ("=" * 78)
    & $Python -m pdbenergy.cli @CliArgs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "stage '$Name' failed with exit code $LASTEXITCODE (see $log)" }
}

if (-not $SkipDownload) {
    $dl = @("download") + $common + @("--force")
    Invoke-Stage -Name "01_download" -CliArgs $dl
    Invoke-Stage -Name "02_inventory" -CliArgs (@("inventory") + $common)
}

if (-not $SkipEnsemble) {
    Invoke-Stage -Name "03_ensemble" -CliArgs (@("ensemble") + $common + @("--threads", $Threads))
}

Invoke-Stage -Name "04_dataset" -CliArgs (@("dataset") + $common)

if (-not $SkipTrain) {
    Invoke-Stage -Name "05_train_schnet" -CliArgs (@("train") + $common + @("--model", "schnet", "--tag", "schnet_protein"))
    Invoke-Stage -Name "06_train_mlp"    -CliArgs (@("train") + $common + @("--model", "mlp",    "--tag", "mlp_protein"))
}

Invoke-Stage -Name "07_eval_schnet" -CliArgs (@("evaluate") + $common + @("--run-dir", "outputs/schnet_protein"))
Invoke-Stage -Name "08_eval_mlp"    -CliArgs (@("evaluate") + $common + @("--run-dir", "outputs/mlp_protein"))
Invoke-Stage -Name "09_ablation"    -CliArgs (@("ablate")   + $common + @("--epochs", 40))

$predictArgs = @("predict") + $common + @("--checkpoint", "outputs/schnet_protein/checkpoint.pt",
                                          "--max-models", 3, "--json", "outputs/predictions.json")
if ($Verify) { $predictArgs += "--verify" }
Invoke-Stage -Name "10_predict" -CliArgs ($predictArgs + @($PredictPdb))

Write-Host ""
Write-Host "Done. Look in outputs\ for checkpoints, metrics.json, metrics_full.json and PNG figures."
