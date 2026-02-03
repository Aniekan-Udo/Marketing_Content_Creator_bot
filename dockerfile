# ===================================================================
# Multi-stage Dockerfile for Marketing Content Creator Bot
# Optimized for CrewAI and heavy ML dependencies
# ===================================================================

# Stage 1: Builder - Install all dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies needed for compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python packages to user directory (for copying to runtime stage)
RUN pip install --no-cache-dir --user -r requirements.txt

# ===================================================================
# Stage 2: Runtime - Slim image with only necessary components
# ===================================================================
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y \
    postgresql-client \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create directories for brand documents if they don't exist
RUN mkdir -p brand_blogs brand_social brand_ads static

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application with uvicorn
# Note: Render/Railway will set the PORT environment variable
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1