FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    wget \
    nano \ 
    libglib2.0-0 \
    python3 \
    python3-dev \
    python3-pip \
    freeglut3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    gdb \
    lldb \
    clang \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    libopencv-dev

RUN git clone https://github.com/tauzn-clock/NYUv2Revisit

WORKDIR /NYUv2Revisit
RUN pip3 install -r ./requirements.txt --break-system-packages