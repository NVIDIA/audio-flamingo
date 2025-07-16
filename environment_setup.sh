#!/usr/bin/env bash
set -e

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.14 -y
    conda activate $CONDA_ENV
    # This is optional if you prefer to use built-in nvcc
    conda install -c nvidia cuda-toolkit -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# Using uv to speedup installations
pip install uv
alias uvp="uv pip"

echo "[INFO] Using python $(which python)"
echo "[INFO] Using pip $(which pip)"
echo "[INFO] Using uv $(which uv)"

# This is required to enable PEP 660 support
pip install --upgrade pip setuptools
pip install -e ".[train,eval]"

pip install hydra-core loguru Pillow pydub
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install FlashAttention2
pip install flash_attn==2.7.3
pip install transformers==4.46.0
pip install pytorchvideo==0.1.5
pip install deepspeed==0.15.4
pip install accelerate==0.34.2
pip install numpy==1.26.4
pip install opencv-python-headless==4.8.0.76
pip install matplotlib
# numpy introduce a lot dependencies issues, separate from pyproject.yaml


# audio
pip install soundfile librosa openai-whisper ftfy
pip install ffmpeg
pip install jiwer
pip install wandb
pip install kaldiio
pip install peft==0.14.0
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')

# Downgrade protobuf to 3.20 for backward compatibility
pip install protobuf==3.20.*

# Replace transformers and deepspeed files
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/

# Quantization requires the newest triton version, and introduce dependency issue
pip install triton==3.1.0 # we don't need this version if we do not use FP8LinearQwen2Config, QLlavaLlamaConfig, etc. It is not compatible with mamba-ssm.