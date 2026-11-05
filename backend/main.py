
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import engine, Base
from auth import router as auth_router
from chatbot import generate_answer


# ==========================================
# DATABASE INITIALIZATION
# ==========================================

Base.metadata.create_all(bind=engine)


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="TweetSense AI",
    description="AI-powered product opinion chatbot using tweets",
    version="1.0.0"
)


# ==========================================
# CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# CHAT REQUEST SCHEMA
# ==========================================

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500,
        description="User's product-related question"
    )


# ==========================================
# HOME ROUTE
# ==========================================

@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Welcome to TweetSense AI API"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "running",
        "application": "TweetSense AI"
    }


# ==========================================
# AUTHENTICATION ROUTES
# ==========================================

app.include_router(auth_router)


# ==========================================
# CHATBOT ROUTE
# ==========================================

@app.post("/chat", tags=["Chatbot"])
def chat(data: ChatRequest):

    try:
        result = generate_answer(data.question)

        return {
            "status": "success",
            "question": data.question,
            "answer": result["answer"],
            "sources": result.get("sources", [])
        }

    except Exception as e:
        print("Chatbot API Error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing your question."
        )
