FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

ENV CREWAI_DISABLE_LITELLM=1
ENV OPENAI_API_KEY=sk-nope

# Expose port (Railway uses $PORT env var)
EXPOSE 8000

# Start server (shell form - Railway compatible)
CMD uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
