# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && \
    echo "${TZ}" > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        dvipng \
        git \
        git-lfs \
        g++ \
        inkscape \
        libcupti-dev \
        libcurl4-openssl-dev \
        libfreetype6-dev \
        libjpeg-dev \
        libpng-dev \
        libsm6 \
        libxext6 \
        libxrender1 \
        libzmq3-dev \
        locales \
        lmodern \
        pkg-config \
        swig \
        texlive-fonts-extra \
        texlive-fonts-recommended \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-latex-recommended \
        texlive-xetex \
        unzip \
        wget \
        zlib1g-dev && \
    locale-gen en_US.UTF-8 && \
    git lfs install --system && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade \
        pip==24.0 \
        setuptools==69.5.1 \
        wheel==0.43.0 && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]