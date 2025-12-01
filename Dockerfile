# =============================================================================
# Deepfake Detection System - Dockerfile
# =============================================================================
# Multi-stage build for optimized production image
#
# Build:
#   docker build -t deepfake-detector .
#   docker build --target dev -t deepfake-detector:dev .
#
# Run:
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models deepfake-detector
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with system dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app


# -----------------------------------------------------------------------------
# Stage 2: Builder - Install Python dependencies
# -----------------------------------------------------------------------------
FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt


# -----------------------------------------------------------------------------
# Stage 3: Development image
# -----------------------------------------------------------------------------
FROM base as dev

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Default command for development
CMD ["python", "-m", "pytest", "tests/"]


# -----------------------------------------------------------------------------
# Stage 4: Production image
# -----------------------------------------------------------------------------
FROM base as production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only necessary files
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY setup.py pyproject.toml ./

# Install package
RUN pip install -e .

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# Default command
ENTRYPOINT ["python"]
CMD ["scripts/detect.py", "--help"]


# -----------------------------------------------------------------------------
# Stage 5: GPU-enabled image (optional)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as gpu

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy requirements and install with GPU support
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install faiss-gpu

# Copy source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY setup.py pyproject.toml ./

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p /app/data /app/models /app/logs

# Default command
ENTRYPOINT ["python"]
CMD ["scripts/detect.py", "--help"]
