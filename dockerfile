FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# FIXED: Use JSON array syntax (exec form)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}", "--workers", "1"]
