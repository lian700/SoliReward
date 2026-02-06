#!/bin/bash
# SoliReward Environment Setup Script
# This script installs all dependencies required for the SoliReward framework

set -e

echo "=== SoliReward Environment Setup ==="

# Create conda environment
echo "Creating conda environment 'solireward'..."
conda create -n solireward python=3.10 -y
conda activate solireward

# Install PyTorch with CUDA 12.4 support
echo "Installing PyTorch..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
echo "Installing core dependencies..."
pip install transformers==4.56.2 trl==0.22.2 datasets==4.1.0 timm==1.0.19 
pip install qwen-vl-utils==0.0.11 math_verify==0.5.2 decord==0.6.0

# Install DeepSpeed for distributed training
echo "Installing DeepSpeed..."
pip install deepspeed==0.16.9

# Install Flash Attention
# Check CXX11 ABI compatibility first
echo "Checking Flash Attention compatibility..."
python -c "import torch; print('CXX11_ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)"

# Install pre-built Flash Attention wheel (for CUDA 12, PyTorch 2.5, CXX11 ABI=False)
# Find other versions at: https://github.com/Dao-AILab/flash-attention/releases
echo "Installing Flash Attention..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install visualization and analysis dependencies
echo "Installing visualization dependencies..."
pip install matplotlib scikit-learn seaborn tensorboard orjson

echo "=== Environment setup complete! ==="
echo "To activate the environment, run: conda activate solireward"
