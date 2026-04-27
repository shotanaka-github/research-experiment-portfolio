#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./setup_env.sh <project-name>"
  echo "Example: ./setup_env.sh krrpca_fried_egg"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/$1"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "Project not found: $PROJECT_DIR"
  exit 1
fi

if [[ ! -f "$PROJECT_DIR/requirements.txt" ]]; then
  echo "requirements.txt not found in: $PROJECT_DIR"
  exit 1
fi

python3 -m venv "$PROJECT_DIR/.venv"
source "$PROJECT_DIR/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$PROJECT_DIR/requirements.txt"

echo
echo "Environment ready:"
echo "  source \"$PROJECT_DIR/.venv/bin/activate\""
