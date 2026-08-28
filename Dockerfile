# 1. Base Image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Create data directory inside container
RUN mkdir -p /app/data

# 2. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt-get/lists/*

# 3. Copy Manifest
COPY requirements.txt .

# 4. INSTALLATION (With Debugging)
# We 'cat' the file to the build logs so you can SEE if pandas is listed.
RUN echo "===== CHECKING REQUIREMENTS =====" && \
    cat requirements.txt && \
    echo "=================================" && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
COPY . .

# 6. Runtime Config
ENV PYTHONPATH=/app/src
CMD ["python", "src/train.py"]