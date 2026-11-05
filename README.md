# TweetSense AI – Tweet-Based Product Opinion Chatbot

An AI-powered chatbot that analyzes product opinions from Twitter/X datasets and generates meaningful answers using semantic search and Large Language Models (LLMs).

Users can ask product-related questions and receive AI-generated insights supported by relevant tweets.

---

## 🚀 Project Overview

TweetSense AI helps users understand what people think about products by analyzing tweet-based opinions.

Instead of manually searching through thousands of tweets, users can simply ask questions such as:

- Is Dell laptop good?
- What are the advantages of HP laptops?
- What do people think about ASUS laptops?

The application retrieves relevant tweets using semantic similarity and uses an LLM to generate a concise, balanced response based on the retrieved information.

## ✨ Key Features

- **User Registration & Login:** Create an account and securely store passwords using hashing.
- **Tweet Dataset Integration:** Load and preprocess product-related tweets from Hugging Face.
- **Semantic Search:** Retrieve relevant tweets using Sentence Transformers and FAISS.
- **AI-Powered Responses:** Generate contextual answers using Llama 3.1 through Ollama.
- **Supporting Tweets:** Display the tweets used to generate each response.
- **REST API:** Backend services developed using FastAPI.
- **Interactive Frontend:** Responsive user interface developed using React and Tailwind CSS.

## 🏗️ System Architecture

```text
              USER
                |
                v
         React Frontend
                |
                v
          FastAPI Backend
                |
        +-------+-------+
        |               |
        v               v
   Authentication   Chatbot API
        |               |
        v               v
   SQLite Database  FAISS Search
                        |
                        v
                 Relevant Tweets
                        |
                        v
                  Context Builder
                        |
                        v
                 Ollama API
                        |
                        v
                 Llama 3.1 LLM
                        |
                        v
                Generated Answer
                        |
                        v
              Supporting Tweets
                        |
                        v
                 React Frontend
```

**LLM Hosting:** Ollama runs in Google Colab, with its API exposed through an ngrok tunnel. The backend connects using the configurable Ollama API URL.

## 🛠️ Technology Stack

### Frontend
- React.js
- JavaScript
- Tailwind CSS
- Vite

### Backend
- Python
- FastAPI
- SQLAlchemy
- SQLite
- Pydantic

### AI / Machine Learning
- Hugging Face Datasets
- Sentence Transformers
- `sentence-transformers/all-MiniLM-L6-v2`
- FAISS
- Ollama
- Llama 3.1

### Tools & Platforms
- Google Colab
- NVIDIA T4 GPU (when available)
- ngrok
- Git & GitHub
- VS Code
- Postman / Swagger UI

## 📂 Project Structure

```text
Tweet-Sense-AI/
│
├── backend/
│   ├── data/
│   │   ├── tweets.json
│   │   ├── tweets.index
│   │   └── tweets_metadata.json
│   │
│   ├── main.py
│   ├── auth.py
│   ├── chatbot.py
│   ├── database.py
│   ├── models.py
│   ├── dataset.py
│   ├── embeddings.py
│   ├── search.py
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
│   │
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── .gitignore
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/TweetSense-AI.git

cd TweetSense-AI
```

### 2. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file inside the backend directory.

```env
HF_TOKEN=your_huggingface_token

OLLAMA_BASE_URL=https://your-ngrok-url.ngrok-free.app

OLLAMA_MODEL=llama3.1:latest
```

Keep your API tokens and private configuration out of GitHub.

### 4. Prepare the Dataset and FAISS Index

If the prepared dataset and index files are not included in the repository, run:

```bash
python dataset.py
python embeddings.py
```

This downloads and preprocesses the dataset, generates embeddings, and creates the FAISS search index.

### 5. Start the Backend

```bash
uvicorn main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Frontend Setup

Open another terminal and navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

## 🔄 How It Works

1. The user registers or logs in to TweetSense AI.
2. The user asks a product-related question.
3. FastAPI receives the question through the chatbot endpoint.
4. Sentence Transformers converts the question into an embedding.
5. FAISS searches for the most relevant tweet embeddings.
6. The retrieved tweets are added to the LLM prompt.
7. Ollama sends the prompt to Llama 3.1.
8. The LLM generates an answer based on the retrieved tweet context.
9. The frontend displays the AI response and supporting tweets.

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Welcome endpoint |
| GET | `/health` | Backend health check |
| POST | `/auth/register` | Register a new user |
| POST | `/auth/login` | Authenticate user |
| POST | `/chat` | Generate an AI-powered product opinion response |

## 📊 Dataset Information

**Dataset:** [Twitter Laptop Dataset – Hugging Face](https://huggingface.co/datasets/MrbBakh/Twitter_Laptop)

The dataset contains laptop-related tweets.

Preprocessing includes:
- Text cleaning
- English-language filtering
- Duplicate removal
- Tweet selection
- Metadata extraction

The processed tweets are stored as JSON and indexed using FAISS.

**Note:** The current dataset focuses on laptops, not all product categories. Retrieved tweets can include advertisements and limited opinions, so AI-generated conclusions should be interpreted accordingly.

## 🔐 Security Notes

- Passwords are stored using password hashing.
- Environment variables are excluded from version control.
- Hugging Face and ngrok credentials should remain private.

The current authentication implementation provides basic registration and login; persistent JWT/session authentication is not yet implemented.

## 🚧 Development Progress

- [x] Project setup and environment configuration
- [x] FastAPI backend setup
- [x] User registration and login
- [x] Hugging Face dataset integration
- [x] Tweet preprocessing
- [x] Embedding generation
- [x] FAISS semantic search
- [x] Ollama LLM integration code
- [x] React frontend setup
- [x] Login and registration interface
- [x] Chatbot interface
- [ ] End-to-end remote LLM integration testing
- [ ] Final frontend and backend integration testing

## 🔮 Future Enhancements

- Support additional product categories.
- Improve sentiment analysis.
- Add conversation history.
- Improve tweet filtering and relevance.
- Add persistent authentication.
- Deploy the application online.

## 👨‍💻 Author

**Hardik Dhamel**

Computer Engineering | Python | AI/ML | Generative AI

---

*TweetSense AI – Turning social media conversations into meaningful product insights.*
