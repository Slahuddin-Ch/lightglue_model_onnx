# Use a minimal Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies one by one
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir onnxruntime
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir opencv-python-headless

# Define environment variable
ENV NAME LightGlueApp

# Run infer_using_CPU.py when the container launches
CMD ["python", "infer_using_CPU.py"]
