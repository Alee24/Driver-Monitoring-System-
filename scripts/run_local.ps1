#!/usr/bin/env pwsh
# Run the driver monitoring system locally on Windows (PowerShell).
#
# This script sets up a Python virtual environment, installs dependencies
# and launches the application.  Additional parameters are passed through
# to the Python module.

Param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [String[]] $Args
)

$base = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $base "..\.venv"

if (-Not (Test-Path $venv)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venv
}

& "$venv\Scripts\activate.ps1"
pip install --upgrade pip | Out-Null
pip install -r (Join-Path $base "..\requirements.txt") | Out-Null

python -m src.dms.app @Args