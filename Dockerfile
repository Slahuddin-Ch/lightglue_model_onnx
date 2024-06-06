# Use a minimal Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir onnxruntime matplotlib numpy opencv-python-headless

# Define environment variable
ENV NAME LightGlueApp

# Run batch_processing.py when the container launches
CMD ["python", "batch_processing.py"]

