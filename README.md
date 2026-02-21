# BrandGuard AI: Marketing Content Creator Bot

BrandGuard AI is a production-ready, multi-agent marketing content generation system designed for speed, simplicity, and brand precision. It leverages an **open-access, frictionless model** that allows you to start generating brand-aligned content instantly without registration or login.

## 🚀 Key Features

- **Frictionless Workflow**: No login required. Launch the dashboard and start building your brand knowledge base immediately.
- **Multi-Agent Architecture**: Powered by CrewAI, utilizing specialized agents (Researcher, Strategist, Analyst, Writer, Reviewer) for a complete content lifecycle.
- **RAG for Brand Alignment**: Integrated with LlamaIndex to ground every generation in your actual brand documents (PDF, DOCX, TXT, MD).
- **Self-Learning Loop**: Agents autonomously adapt to your style based on human approvals and feedback stored in a dedicated learning memory.
- **Premium Dark UI**: A high-performance, minimalist dashboard designed for professional marketing workflows.
- **Production Resilience**:
  - **Circuit Breakers**: Protects against cascading failures in external APIs (Groq, Tavily).
  - **Structured Logging**: Production-grade observability with `structlog`.
  - **Self-Healing Infrastructure**: Automatic directory management and database health monitoring.

## 🛠️ Tech Stack

- **Framework**: FastAPI (Python 3.12+)
- **Orchestration**: CrewAI & LlamaIndex
- **LLM**: Groq (Llama-3-70b/8b)
- **Database**: PostgreSQL (SQLAlchemy)
- **Styling**: Vanilla CSS + Tailwind (Premium Dark Mode)

---

## 💻 Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) (Preferred) or Python 3.12+
- PostgreSQL
- Groq API Key
- Tavily API Key (Optional)

### Quick Start

1. **Clone & Setup**:
   ```bash
   git clone <repo-url>
   cd marketing-automation-agent
   ```

2. **Configure Environment**:
   Create a `.env` file:
   ```env
   POSTGRES_URI=postgresql://user:password@localhost:5432/brandguard
   GROQ_API_KEY=your_groq_key
   TAVILY_API_KEY=your_tavily_key
   ENVIRONMENT=development
   ```

3. **Run Application**:
   Using `uv` (recommended):
   ```bash
   uv run app.py
   ```
   Or standard python:
   ```bash
   python app.py
   ```

---

## 🌐 Deployment (Render)

This project includes a `render.yaml` for one-click deployment to **Render**.

1. Connect your GitHub repository to Render.
2. Render will automatically detect the blueprint and provision:
   - **Web Service**: The FastAPI application.
   - **PostgreSQL**: The database.
3. Configure your `GROQ_API_KEY` and `TAVILY_API_KEY` in the Render dashboard environment settings.

---

## 📡 API Reference

- **`GET /`**: Premium Landing Page
- **`GET /app`**: Frictionless Dashboard
- **`POST /api/upload`**: Build Knowledge Base from files
- **`POST /api/generate`**: Start Multi-Agent Pipeline
- **`POST /api/feedback`**: Submit Human Feedback (Trains the system)
- **`GET /health`**: System & Circuit Breaker Health

---

## 🛡️ Security & Reliability

The system is built to handle production loads with:
- **Error Masking**: Genericized public errors to prevent stack trace leakage.
- **Health Checks**: Real-time monitoring of database and LLM provider connectivity.
- **Safe Fallbacks**: Graceful handling of missing brand data or API timeouts.

## 📜 License

[MIT License](LICENSE)
