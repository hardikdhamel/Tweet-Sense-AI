# TweetSense AI – AI-Powered Product Opinion Chatbot

TweetSense AI is a web-based chatbot that helps users understand product opinions by asking questions in natural language. It retrieves relevant posts from a curated Twitter/X dataset using semantic search and generates concise, readable answers with Google Gemini.

<p align="center">
  <a href="https://tweetsense-ai.web.app">Live Demo</a> ·
  <a href="https://tweet-sense-ai.onrender.com/docs">API Documentation</a>
</p>

---

## Overview

Instead of manually searching through a large collection of product-related posts, users can ask questions such as:

- Is Dell laptop good?
- What are the advantages of HP laptops?
- What do people think about ASUS laptops?

TweetSense AI finds relevant content in its indexed dataset and uses that retrieved context to generate an answer. The current dataset is focused on laptop-related posts.

## Features

- **User registration and login** with password hashing.
- **Natural-language product questions** through an interactive chat interface.
- **Semantic retrieval** using Jina embeddings and FAISS vector search.
- **Retrieval-augmented generation (RAG)** with Google Gemini.
- **Concise, structured answers** with readable Markdown formatting.
- **Responsive React interface** with a chat experience.
- **FastAPI REST backend** with API documentation.
- **Cloud deployment** with Firebase Hosting and Render.
- **PostgreSQL in production** using Neon; SQLite is available as the local fallback configured by the backend.

> The chat interface displays the generated answer rather than listing the retrieved posts. The retrieval results are used internally as context for answer generation.

## Live Application

| Component | URL |
|---|---|
| Frontend | https://tweetsense-ai.web.app |
| Backend API | https://tweet-sense-ai.onrender.com |
| API documentation | https://tweet-sense-ai.onrender.com/docs |
| Health check | https://tweet-sense-ai.onrender.com/health |

## System Architecture

```text
                         USER
                           |
                           v
                  React + Vite Frontend
                    (Firebase Hosting)
                           |
                           v
                    FastAPI Backend
                       (Render)
                           |
              +------------+-------------+
              |                          |
              v                          v
       Authentication                Chatbot API
              |                          |
              v                          v
       Neon PostgreSQL             Jina Embeddings API
       (SQLite locally)                   |
                                          v
                                  Query Embedding
                                          |
                                          v
                                    FAISS Index
                                          |
                                          v
                                Top 5 Relevant Posts
                                          |
                                          v
                                  Context Builder
                                          |
                                          v
                                    Gemini API
                                          |
                                          v
                                  Final Answer
                                          |
                                          v
                                  React Chat UI
```

## How It Works

1. A user registers or logs in.
2. The user submits a product-related question in the chat interface.
3. The React frontend sends the question to the FastAPI `/chat` endpoint.
4. The backend sends the question to Jina to generate a query embedding.
5. FAISS searches the pre-built vector index and retrieves the five most relevant posts.
6. The backend builds a context from the retrieved results.
7. Google Gemini generates an answer using the question and retrieved context.
8. The frontend renders the answer with Markdown formatting.

## Technology Stack

### Frontend
- React 19
- JavaScript
- Vite
- Tailwind CSS
- `react-markdown`

### Backend
- Python
- FastAPI
- Uvicorn
- Pydantic
- SQLAlchemy
- SQLite (local fallback)
- PostgreSQL / Neon (production)
- `psycopg`

### AI / Search
- Hugging Face Datasets
- Jina Embeddings API – `jina-embeddings-v5-text-small`
- 1024-dimensional embeddings
- FAISS (`IndexFlatIP`) for vector similarity search
- Google Gemini API – configured through `GEMINI_MODEL`
- Retrieval-Augmented Generation (RAG)

### Deployment and Tools
- Firebase Hosting – frontend
- Render – backend
- Neon – hosted PostgreSQL
- Hugging Face – source dataset
- Git and GitHub
- VS Code
- Swagger UI / Postman

## Dataset

**Source:** [Twitter Laptop Dataset – Hugging Face](https://huggingface.co/datasets/MrbBakh/Twitter_Laptop)

The current application uses a prepared collection of approximately 3,000 laptop-related posts. The data preparation workflow includes text preprocessing, filtering, duplicate removal, and metadata preparation.

The processed data is embedded with Jina and stored in a FAISS index for semantic retrieval. The current release is built around this laptop dataset; support for dynamically selecting or switching Hugging Face datasets is not part of the current implementation.

## Project Structure

The main project structure is:

```text
Tweet-Sense-AI/
│
├── backend/
│   ├── data/
│   │   ├── tweets.json
│   │   ├── tweets.index
│   │   ├── tweets_metadata.json
│   │   ├── tweets_jina.index
│   │   └── tweets_jina_metadata.json
│   │
│   ├── main.py
│   ├── auth.py
│   ├── chatbot.py
│   ├── dataset.py
│   ├── embeddings.py
│   ├── embeddings_jina.py
│   ├── search.py
│   ├── compare_embeddings.py
│   ├── database.py
│   ├── models.py
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── firebase.json
│
├── .gitignore
└── README.md
```

The exact files may vary as the project evolves. Generated vector indexes and metadata files must be present in `backend/data/` for the current search implementation.

## Run Locally

### Prerequisites

- Python 3.10 or later
- Node.js and npm
- API keys for Jina and Google Gemini
- A PostgreSQL connection string if you want to use PostgreSQL locally; otherwise, use the configured SQLite fallback

### 1. Clone the repository

```bash
git clone https://github.com/hardikdhamel/Tweet-Sense-AI.git
cd Tweet-Sense-AI
```

### 2. Configure the backend

Open a terminal in the project root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file inside `backend/`:

```env
# Required AI service credentials
JINA_API_KEY=your_jina_api_key
GEMINI_API_KEY=your_gemini_api_key

# Model configuration
JINA_EMBEDDING_MODEL=jina-embeddings-v5-text-small
GEMINI_MODEL=gemini-3.5-flash-lite

# Database
# Set this to your PostgreSQL connection string when using PostgreSQL.
# If omitted, the backend uses its configured SQLite fallback.
DATABASE_URL=

# Frontend origin allowed by the backend CORS configuration
FRONTEND_URL=http://localhost:5173
```

Do not commit `.env` or publish API keys, database credentials, or other secrets.

### 3. Ensure the search data is available

The current search implementation expects the Jina FAISS index and its metadata in:

```text
backend/data/tweets_jina.index
backend/data/tweets_jina_metadata.json
```

If you need to recreate the prepared data, run the project's data preparation and Jina indexing scripts from the `backend/` directory:

```powershell
python dataset.py
python embeddings_jina.py
```

This step requires the dataset source and a valid Jina API key. If the prepared index files are already present, you do not need to rebuild them.

### 4. Start the backend

From the `backend/` directory:

```powershell
uvicorn main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

### 5. Configure and start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
```

Create `frontend/.env` for local development:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Start Vite:

```powershell
npm run dev
```

Frontend:

```text
http://localhost:5173
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API welcome response |
| `GET` | `/health` | Backend health check |
| `POST` | `/auth/register` | Register a user |
| `POST` | `/auth/login` | Log in a user |
| `POST` | `/chat` | Retrieve relevant posts and generate an answer |

See `/docs` for the request and response schemas exposed by the running API.

## Deployment

### Frontend – Firebase Hosting

The React/Vite frontend is deployed to Firebase Hosting:

```text
https://tweetsense-ai.web.app
```

For a production build, configure the frontend API URL before building:

```env
VITE_API_URL=https://tweet-sense-ai.onrender.com
```

Then build and deploy from the `frontend/` directory:

```bash
npm run build
firebase deploy --only hosting
```

Vite environment variables are included at build time. Rebuild and redeploy the frontend after changing `VITE_API_URL`.

### Backend – Render

The FastAPI backend is deployed on Render:

```text
https://tweet-sense-ai.onrender.com
```

The Render service needs the backend environment variables configured in its dashboard, including the Jina and Gemini API keys, model names, database URL, and allowed frontend URL.

Set:

```env
FRONTEND_URL=https://tweetsense-ai.web.app
```

Do not place secret values in this README or in source control.

### Database – Neon

The deployed backend uses a Neon PostgreSQL database through `DATABASE_URL`. Keep the connection string private and configure it as an environment variable in Render.

## Security

- Passwords are stored using password hashing.
- API keys and database connection strings are loaded from environment variables.
- `.env` files and local secrets should be excluded from Git.
- Use HTTPS for deployed frontend and backend communication.
- Never commit real API keys, database passwords, or user credentials.

## Current Scope and Future Improvements

The current release focuses on answering questions from the prepared laptop-related dataset. Potential future improvements include:

- Support for additional product categories and configurable datasets.
- Better filtering and ranking of retrieved content.
- More robust evaluation of retrieval relevance and answer quality.
- Conversation history.
- Further authentication and account-management improvements.

These are future enhancements and should not be considered implemented features.

## Author

**Hardik Dhamel**  
Computer Engineering | Python | AI/ML | Generative AI

---

*TweetSense AI – Turning product conversations into useful insights.*
