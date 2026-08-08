# Dockerfile to run rayfronts
# Run with --gpus all --network host --ipc host --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all 

FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y

## Install ROS2 Humble
RUN apt update && apt install locales && locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && export LANG=en_US.UTF-8

RUN apt install software-properties-common -y && add-apt-repository universe && \
 apt update && apt install curl -y && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && apt update
 
RUN apt install ros-humble-desktop -y
# Detection2DArray for the people-detection subscriber
RUN apt install ros-humble-vision-msgs -y
# Cross-domain bridge for the MAIPP robot_1 <-> robot_2 coordination topics
RUN apt install ros-humble-domain-bridge -y

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Setup python utils
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-requests \
    python3-yaml \
    wget

## Install python dependencies
# torch >= 2.7 + cu128 needed for Blackwell GPUs (RTX 50xx, sm_120)
RUN pip install \
  torch==2.7.1 \
  torchvision==0.22.1 \
  torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

# torch-scatter ships compiled kernels tied to the torch version above;
# install from the matching PyG wheel index
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html

RUN pip install \
  protobuf \
  onnx \
  scipy==1.15.2 \
  rerun-sdk==0.22.0 \
  einops \
  timm \
  ftfy \
  regex \
  nanobind \
  hydra-core \
  open_clip_torch \
  transformers \
  idna==3.10 \
  requests==2.32.3 \
  pandas

## Compile & install patched open-vdb (Need open-vdb 12.0 that exposes Int8Grid to python)

# Get > 1.8 boost version for compatibility with openvdb 12.0 
# But need to downgrade numpy as latest boost 1.86 not compatible with numpy 2
RUN pip install "numpy<2"
RUN wget https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz && \
  tar xzvf boost_1_86_0.tar.gz && cd boost_1_86_0 && ./bootstrap.sh --prefix=/usr && ./b2 && ./b2 install


WORKDIR /workspace
RUN apt-get update && apt-get install -y git && \ 
rm -rf /var/lib/apt/lists/* && \ 
git clone https://github.com/OasisArtisan/openvdb && mkdir openvdb/build

WORKDIR /workspace/openvdb/build
RUN apt-get install -y cmake libboost-iostreams-dev libtbb-dev libblosc-dev python3-dev python3-numpy python3-pip
RUN pip3 install nanobind
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DOPENVDB_BUILD_PYTHON_MODULE=ON -DOPENVDB_BUILD_PYTHON_UNITTESTS=ON \
  -DOPENVDB_PYTHON_WRAP_ALL_GRID_TYPES=ON -DUSE_NUMPY=ON \
  -Dnanobind_DIR=/usr/local/lib/python3.10/dist-packages/nanobind/cmake ..

RUN make -j4
RUN make install

# transformers 5.x needs PIL.Image.Resampling (Pillow >= 9.1); the system
# python3-pil package is 9.0.1
RUN pip install -U pillow

# Clone rayfronts and compile
WORKDIR /workspace
RUN git clone https://github.com/RayFronts/RayFronts.git
RUN cd RayFronts && CMAKE_INSTALL_PREFIX=/usr/local ./compile.sh

WORKDIR /
