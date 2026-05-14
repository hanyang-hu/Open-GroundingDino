# Dockerfile for Fine-Tuning Grounding DINO
# Base image: NVIDIA CUDA with cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /workspace/grounding_dino

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    vim \
    wget \
    curl \
    tmux \
    htop \
    unzip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install basic tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements file for better Docker layer caching
COPY requirements.txt /workspace/grounding_dino/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the project code
COPY . /workspace/grounding_dino/

# Set Python to unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Default command: bash shell
CMD ["/bin/bash"]
