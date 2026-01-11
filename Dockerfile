# Base image with Python and CUDA support
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg git git-lfs && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir runpod huggingface_hub onnxruntime soundfile numpy librosa PyYAML

# Initialize git-lfs
RUN git lfs install

# Download ONNX models from official HuggingFace repo
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Supertone/supertonic-2', local_dir='/app/assets', allow_patterns=['onnx/*', 'voice_styles/*'])"

# Clone and copy Supertonic Python files (avoid symlinks)
RUN git clone --depth 1 --filter=blob:none --sparse https://github.com/supertone-inc/supertonic.git /tmp/supertonic && \
    cd /tmp/supertonic && \
    git sparse-checkout set py && \
    cp /tmp/supertonic/py/*.py /app/ && \
    rm -rf /tmp/supertonic

# Copy the handler
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the handler
CMD ["python", "-u", "handler.py"]
