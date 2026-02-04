# BrandGuard AI: Marketing Content Creator Bot

BrandGuard AI is a production-ready, multi-agent marketing content generation system designed to create brand-aligned content that improves over time through human feedback. It leverages RAG (Retrieval-Augmented Generation) to ground generation in your specific brand voice and a self-learning loop to adapt to your style.

## Key Features

- **Multi-Agent Architecture**: Powered by CrewAI, utilizing specialized agents for research, writing, and review.
- **RAG for Brand Alignment**: Integrated with LlamaIndex to retrieve context from uploaded brand documents (PDF, DOCX, TXT).
- **Self-Learning Feedback Loop**: Automatically adapts to brand style based on human approvals and feedback.
- **Production Resilience**:
  - **Circuit Breakers**: Protects against cascading failures in external APIs (Groq, Tavily).
  - **Rate Limiting**: Integrated token-bucket limits for API protection.
  - **Retry Logic**: Exponential backoff for database and API operations.
  - **Persistent Storage**: All brand documents are stored in PostgreSQL for persistence across deployments.
- **Real-time Metrics**: Analyzes brand tone, sentence structure, and signature phrases.

## Tech Stack

- **Framework**: FastAPI (Python)
- **AI/LLM**: Groq (Llama-3), CrewAI, LlamaIndex
- **Database**: PostgreSQL (SQLAlchemy + psycopg2)
- **Monitoring**: Structlog (Structured Logging)
- **Deployment**: Railway

---

## Getting Started (Local Development)

### Prerequisites

- Python 3.10+
- PostgreSQL
- Groq API Key
- Tavily API Key (Optional, for web search capabilities)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd <project-directory>
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   POSTGRES_URI=postgresql://user:password@localhost:5432/brandguard
   GROQ_API_KEY=your_groq_key
   TAVILY_API_KEY=your_tavily_key
   ENVIRONMENT=development
   ```

5. **Initialize the database**:
   ```bash
   python -c "from db import init_db; init_db()"
   ```

6. **Run the application**:
   ```bash
   uvicorn app:app --reload
   ```

---

## Deployment to Railway

This project is optimized for [Railway](https://railway.app).

### 1. Create a New Project
In the Railway dashboard, click **New Project** and select **Deploy from GitHub repo**.

### 2. Add PostgreSQL Service
Click **+ Add Service** -> **Database** -> **Add PostgreSQL**. Railway will automatically provision a database and provide a `DATABASE_URL`.

### 3. Configure Environment Variables
In your main application service (the one linked to your GitHub repo), add the following variables:

| Variable | Description |
| :--- | :--- |
| `POSTGRES_URI` | Copy your Railway PostgreSQL `DATABASE_URL` here |
| `GROQ_API_KEY` | Your Groq API key |
| `TAVILY_API_KEY` | Your Tavily API key |
| `ENVIRONMENT` | Set to `production` |
| `PORT` | Set to `8000` (Railway usually handles this automatically) |

### 4. Database Migration
If you are updating an existing deployment or if it's the first run, ensure you run the migration script to add required columns:

```bash
# Locally (with Railway CLI)
railway run python add_file_content_column.py

# Or wait for the app to start (it calls init_db() automatically on startup)
```

---

## API Reference

### Health Check
`GET /health`
Returns system status, database connectivity, and circuit breaker states.

### Process Flow
1. **Upload Documents**: `POST /api/upload` (Upload brand samples)
2. **Generate Content**: `POST /api/generate` (Create brand-aligned content)
3. **Submit Feedback**: `POST /api/feedback` (Train the system on your preferences)
4. **View Stats**: `GET /api/learning/{business_id}/stats` (Track learning progress)

---

## Monitoring & Health

Access the health dashboard at `/health`. The system logs structured output (JSON in production) for easier log aggregation.

- **Circuit Breaker Status**: Check if external services are failing.
- **Connection Pool**: Monitor database connection usage.
- **Learning Stats**: Verify the AI is accurately predicting your feedback.

## License

[MIT License](LICENSE)