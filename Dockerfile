# 1. Base Image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# Create data directory inside container
RUN mkdir -p /app/data

# 2. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default 'python' and 'pip'
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 3. Copy Manifest
COPY requirements.txt .

# 4. INSTALLATION (With Debugging)
# We 'cat' the file to the build logs so you can SEE if pandas is listed.
RUN echo "===== CHECKING REQUIREMENTS =====" && \
    cat requirements.txt && \
    echo "=================================" && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt

# 5. Copy Application Code
COPY . .

# 6. Runtime Config
ENV PYTHONPATH=/app/src
CMD ["python", "src/train.py"]