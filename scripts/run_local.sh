#!/usr/bin/env bash
# Run the driver monitoring system locally on UNIX‑like systems.
#
# This script sets up a Python virtual environment, installs dependencies
# and launches the application.  Additional command‑line arguments are passed
# through to the Python module.

set -euo pipefail

if [[ ! -d "$(dirname "$0")/../.venv" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "$(dirname "$0")/../.venv"
fi
source "$(dirname "$0")/../.venv/bin/activate"
pip install --upgrade pip >/dev/null
pip install -r "$(dirname "$0")/../requirements.txt" >/dev/null

# Run the app with any provided arguments
python -m src.dms.app "$@"