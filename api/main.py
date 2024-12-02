from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pickle
import sys
import os

# Ensure the project root is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def preprocess(text):
    # Your preprocessing logic here
    return text

def load_models():
    try:
        # Use absolute or relative paths carefully in serverless environments
        vectoriser = pickle.load(open('models/vectoriser-ngram-(1,2).pickle', 'rb'))
        LRmodel = pickle.load(open('models/Sentiment-LRv1.pickle', 'rb'))
        return vectoriser, LRmodel
    except Exception as e:
        print(f"Model loading error: {e}")
        return None, None

def predict(vectoriser, model, texts):
    textdata = vectoriser.transform(preprocess(texts))
    sentiment = model.predict(textdata)
    
    return [
        {"text": text, "sentiment": pred} 
        for text, pred in zip(texts, sentiment)
    ]

# Load models
vectoriser, LRmodel = load_models()

# Create FastAPI app
app = FastAPI(title="Sentiment Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Request model
class TextInput(BaseModel):
    texts: List[str]

# Sentiment prediction endpoint
@app.post("/predict")
async def predict_sentiment(input: TextInput):
    if vectoriser is None or LRmodel is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        results = predict(vectoriser, LRmodel, input.texts)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint for health check
@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API is running"}

# This is crucial for Vercel serverless functions
def create_app():
    return app