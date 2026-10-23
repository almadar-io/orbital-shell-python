# ===========================================
# Orbital Python Shell - Dockerfile
# ===========================================
# Multi-stage build for Cloud Run deployment

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files (src needed for setuptools to find packages)
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code and the native kernel extension (kernel/)
COPY src/ ./src/
COPY kernel/ ./kernel/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production
ENV NODE_ENV=production
ENV ORBITAL_KERNEL_DIR=/app/kernel

# Port (default 3030, Cloud Run overrides via PORT env var)
ENV PORT=3030

# Expose port
EXPOSE 3030

# Run the application (the served program listens on $PORT)
CMD ["python", "-m", "orbital_app"]
