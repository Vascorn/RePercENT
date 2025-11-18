# 1. Base Image (Includes Python 3.10, CUDA, PyTorch)
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# 2. Install system tools (git, curl, etc.)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy your ingredients list first (for caching)
COPY requirements.txt .

# 5. Install the libraries from your list
RUN pip install --no-cache-dir -r requirements.txt

# 6. Keep the container alive so you can connect via VS Code
CMD ["sleep", "infinity"]