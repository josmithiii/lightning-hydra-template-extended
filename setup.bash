#!/usr/bin/env bash

# Setup script for nsmt project using uv
# Creates virtual environment and installs dependencies

set -e  # Exit on any error

echo "Setting up Neural Spectral Modeling Template project environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment with Python 3.9+
echo "Creating virtual environment..."
uv venv .venv --python 3.9

# Activate environment and install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "Setup complete! Virtual environment created with all dependencies including Hydra."
echo ""
echo "IMPORTANT: You must activate the environment before running any Python commands:"
echo "  source .venv/bin/activate     # For bash/zsh"
echo "  source .venv/bin/activate.csh # For csh/tcsh"
echo ""
echo "Then you can run training commands like:"
echo "  python src/train.py experiment=vimh_cnn"
echo "  make train"
echo ""
echo "To deactivate later, just run: deactivate"
