FROM --platform=linux/amd64 python:3.8-slim-buster as build

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk2.0-dev \
    pkg-config \
    libatlas-base-dev \
    gfortran \
    wget \
    unzip \
    libx11-dev

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Build and install dlib without AVX support
RUN wget http://dlib.net/files/dlib-19.22.tar.bz2 && \
    tar xvjf dlib-19.22.tar.bz2 && \
    cd dlib-19.22 && \
    mkdir build && \
    cd build && \
    cmake .. -DDLIB_USE_CUDA=0 -DUSE_AVX_INSTRUCTIONS=0 -DDLIB_NO_GUI_SUPPORT=1 -DDLIB_DISABLE_ASSERTS=1 && \
    cmake --build . && \
    cd ../.. && \
    pip install dlib-19.22.tar.bz2 && \
    rm -rf dlib-19.22 dlib-19.22.tar.bz2

# Build and install OpenCV without NNPACK support
RUN git clone --branch 4.5.4 https://github.com/opencv/opencv.git && \
    git clone --branch 4.5.4 https://github.com/opencv/opencv_contrib.git && \
    cd opencv && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D WITH_NNPACK=OFF .. && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf opencv opencv_contrib

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "api.py"]
