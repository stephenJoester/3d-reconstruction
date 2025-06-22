FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04

# 1. Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# 2. Install system dependencies
RUN apt-get update && apt-get install -y \
    wget git curl ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libxext6 libsm6 libxrender1 \
    gcc g++ build-essential cmake \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p $CONDA_DIR && \
    rm /miniconda.sh

# 4. Use bash for all commands
SHELL ["/bin/bash", "-c"]

# 5. Create conda env and install dependencies
RUN conda create -n geoenv python=3.10 -y && \
    source activate geoenv && \
    conda install -y -c conda-forge libffi=3.3 && \
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath && \
    conda install -y -c pytorch -c nvidia -c pytorch3d pytorch torchvision torchaudio pytorch-cuda=12.1 pytorch3d

# 6. Install other Python packages via pip
RUN conda run -n geoenv pip install numpy open3d trimesh numba fastapi uvicorn google-cloud-storage pymeshlab

# 7. Set working directory
WORKDIR /workspace

# 8. Copy project files
COPY pointnet2_ops_lib ./pointnet2_ops_lib
COPY weights ./weights
COPY main.py ./
COPY model.py ./
COPY utils.py ./

# 9. Build pointnet2_ops
RUN conda run -n geoenv bash -c "cd pointnet2_ops_lib && python setup.py install"

# 10. Run the application
SHELL ["/bin/bash", "-c"]
RUN echo "source activate geoenv" >> ~/.bashrc
ENV PATH /opt/conda/envs/geoenv/bin:$PATH

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]