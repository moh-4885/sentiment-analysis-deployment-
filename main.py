import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List

# Preprocessing and model loading functions (assuming these are defined elsewhere)
def preprocess(text):
    # Your text preprocessing logic here
    return text

def load_models():
    
    
    # Load the vectoriser.
    file = open('vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    
    # Load the LR Model.
    file = open('Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    
    return vectoriser, LRmodel

def predict(vectoriser, model, texts):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(texts))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(texts, sentiment):
        data.append((text, pred))
    
    return data

# Load models at startup
vectoriser, LRmodel = load_models()

# FastAPI app
app = FastAPI(title="Sentiment Analysis API")

# Request model
class TextInput(BaseModel):
    texts: List[str]

# Response model
class SentimentResult(BaseModel):
    text: str
    sentiment: int

# Sentiment prediction endpoint
@app.post("/predict", response_model=List[SentimentResult])
async def predict_sentiment(input: TextInput):
    try:
        results = predict(vectoriser, LRmodel, input.texts)
        return [
            SentimentResult(text=text, sentiment=sentiment) 
            for text, sentiment in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API (for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)