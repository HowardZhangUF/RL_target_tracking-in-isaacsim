param(
  [int]$Runs = 30
)

$ErrorActionPreference = "Stop"

# --- Fixed paths (as you specified) ---
$IsaacSim   = "C:\isaacsim"
$PFScript   = "C:\isaacsim\standalone_examples\custom_env\ppo_env.py"
$PlotScript = "C:\isaacsim\standalone_examples\custom_env\plot.py"
$GifOutput  = "C:\isaacsim\standalone_examples\custom_env\pf_switch_log_traj.gif"
$OutDir     = "C:\isaacsim\standalone_examples\custom_env\gifs"

function Run-InIsaacSim {
  param([string]$What, [string]$ScriptPath)
  Write-Host ">>> $What..."
  Push-Location $IsaacSim
  try {
    & ".\python.bat" "$ScriptPath"
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
  } finally {
    Pop-Location
  }
}

function Wait-ForFileStable {
  param([string]$Path, [int]$TimeoutSec = 600)
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  $lastSize = -1
  while ((Get-Date) -lt $deadline) {
    if (Test-Path -LiteralPath $Path) {
      try {
        $size = (Get-Item -LiteralPath $Path).Length
        if ($size -gt 0 -and $size -eq $lastSize) { return $true }  # size stopped changing
        $lastSize = $size
      } catch { }
    }
    Start-Sleep -Milliseconds 500
  }
  return $false
}

function Move-WithRetry {
  param([string]$Src, [string]$Dst, [int]$Tries = 10, [int]$DelayMs = 500)
  for ($t=1; $t -le $Tries; $t++) {
    try {
      Move-Item -LiteralPath $Src -Destination $Dst -Force
      return
    } catch {
      if ($t -eq $Tries) { throw "Failed to move $Src to $Dst after $Tries tries. $_" }
      Start-Sleep -Milliseconds $DelayMs
    }
  }
}

# Ensure output folder exists
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

for ($i = 1; $i -le $Runs; $i++) {
  Write-Host "`n===== RUN $i of $Runs ====="

  # 1) Run Isaac Sim (must be invoked from C:\isaacsim with python.bat)
  Run-InIsaacSim -What "Simulation" -ScriptPath $PFScript

  # 2) Run plotting (also from C:\isaacsim)
  Run-InIsaacSim -What "Plotting" -ScriptPath $PlotScript

  # 3) Wait for GIF to be fully written (plot_test prints: [saved] ...pf_switch_log_traj.gif)
  if (-not (Wait-ForFileStable -Path $GifOutput -TimeoutSec 600)) {
    throw "Timed out waiting for GIF: $GifOutput"
  }

  # 4) Rename to vN.gif and loop
  $dest = Join-Path $OutDir ("v{0}.gif" -f $i)
  Move-WithRetry -Src $GifOutput -Dst $dest
  Write-Host "[saved] $dest"
}

Write-Host "`nAll runs completed successfully."
