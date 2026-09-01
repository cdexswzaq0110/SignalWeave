$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    $bootstrapPython = Get-Command python -ErrorAction SilentlyContinue
    if ($bootstrapPython) {
        & $bootstrapPython.Source -m venv (Join-Path $projectRoot '.venv')
    } else {
        py -3 -m venv (Join-Path $projectRoot '.venv')
    }
    & $python -m pip install --upgrade pip
    & $python -m pip install -e "$projectRoot[dev]"
}

Write-Host 'SignalWeave will be available at http://127.0.0.1:8010'
& $python -m signalweave
