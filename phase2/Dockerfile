# Start with an NVIDIA CUDA base image that supports CUDA 12.4
FROM nvidia/cuda:12.4-base-ubuntu20.04

# Install Python 3.9 and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.9 python3-pip python3.9-venv python3.9-dev && \
    ln -s /usr/bin/python3.9 /usr/local/bin/python3 && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install PyTorch, torchvision, torchaudio, torchmetrics
RUN pip install --no-cache-dir torch==2.1.2+cu124 torchvision==0.16.2+cu124 torchaudio==0.13.1+cu124 torchmetrics==1.3.2 -f https://download.pytorch.org/whl/torch_stable.html

# If you have additional dependencies in requirements.txt, install them as well
# RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python3", "train.py"]
