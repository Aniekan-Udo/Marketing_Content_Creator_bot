FROM python:3.11-slim

# Install only runtime deps first (cached layer)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=all -r requirements.txt

# Copy code last (changes don't bust pip cache)
COPY . .

EXPOSE $PORT
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}", "--workers", "1"]
