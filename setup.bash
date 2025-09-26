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

# Check if CUDA is available and install appropriate PyTorch version
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    # Install other requirements (excluding torch packages)
    uv pip install -r requirements.txt --no-deps
    uv pip install lightning>=2.0.0 torchmetrics>=0.11.4 hydra-core==1.3.2 hydra-colorlog==1.2.0 hydra-optuna-sweeper==1.2.0 tensorboard>=2.15.0 rootutils pre-commit rich pytest torchview torchviz matplotlib
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    uv pip install -r requirements.txt
fi

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
