FROM --platform=linux/amd64 python:3.8-slim-buster as build

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    unzip \
    libx11-dev  # Add libx11-dev for X11 support if needed

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install dlib without AVX and NNPACK support
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

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "api.py"]
