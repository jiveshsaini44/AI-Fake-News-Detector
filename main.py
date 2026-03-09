
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re
import asyncio
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import json

app = FastAPI(title="Fake News Detector API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
tokenizer = None
model = None
device = torch.device("cpu")  # Use CPU for i5 laptop

class NewsRequest(BaseModel):
    text: str
    url: str = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

manager = ConnectionManager()

def load_model():
    """Load lightweight model optimized for CPU"""
    global tokenizer, model
    
    try:
        # Using DistilBERT - lighter version of BERT
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using fallback keyword-based detection")

def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()

def extract_article_from_url(url: str) -> str:
    """Extract article text from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Get text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        
        return text if text else soup.get_text()
    except Exception as e:
        print(f"Error extracting URL content: {e}")
        return ""

def keyword_based_detection(text: str) -> Dict:
    """Fallback keyword-based fake news detection"""
    text_lower = text.lower()
    
    # Suspicious keywords that often appear in fake news
    fake_indicators = [
        'shocking', 'unbelievable', 'you won\'t believe',
        'secret they don\'t want you to know', 'doctors hate',
        'click here', 'breaking exclusive', 'must see'
    ]
    
    # Credibility indicators
    credible_indicators = [
        'according to', 'study shows', 'research indicates',
        'expert says', 'data suggests', 'reported by'
    ]
    
    fake_score = sum(1 for keyword in fake_indicators if keyword in text_lower)
    credible_score = sum(1 for keyword in credible_indicators if keyword in text_lower)
    
    # Simple heuristic
    if fake_score > credible_score:
        return {
            'prediction': 'FAKE',
            'confidence': min(0.6 + (fake_score * 0.05), 0.85),
            'method': 'keyword_analysis'
        }
    else:
        return {
            'prediction': 'REAL',
            'confidence': min(0.6 + (credible_score * 0.05), 0.85),
            'method': 'keyword_analysis'
        }

def predict_news(text: str) -> Dict:
    """Predict if news is fake or real"""
    if model is None or tokenizer is None:
        return keyword_based_detection(text)
    
    try:
        # Tokenize (limit to 512 tokens for efficiency)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()
        
        label = "FAKE" if prediction == 0 else "REAL"
        
        return {
            'prediction': label,
            'confidence': float(confidence),
            'method': 'transformer_model',
            'fake_probability': float(probs[0][0]),
            'real_probability': float(probs[0][1])
        }
    except Exception as e:
        print(f"Model prediction error: {e}")
        return keyword_based_detection(text)

def get_fact_check_info(query: str) -> List[Dict]:
    """Get fact-checking information from search (simulated)"""
    # In production, integrate with fact-checking APIs like:
    # - Google Fact Check Tools API
    # - ClaimReview markup
    # For now, return placeholder
    return [
        {
            'source': 'Fact-Check Required',
            'message': 'Please verify this claim through trusted fact-checking websites like Snopes, FactCheck.org, or PolitiFact.'
        }
    ]

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    return {
        "message": "Fake News Detector API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.post("/analyze")
async def analyze_news(request: NewsRequest):
    """Analyze news article via REST API"""
    try:
        text = request.text
        
        # If URL provided, extract content
        if request.url:
            url_text = extract_article_from_url(request.url)
            if url_text:
                text = url_text
        
        # Clean text
        cleaned_text = clean_text(text)
        
        if not cleaned_text or len(cleaned_text) < 20:
            raise HTTPException(status_code=400, detail="Text too short for analysis")
        
        # Predict
        result = predict_news(cleaned_text)
        
        # Get fact-check info if fake
        fact_checks = []
        if result['prediction'] == 'FAKE':
            fact_checks = get_fact_check_info(cleaned_text[:100])
        
        return {
            'status': 'success',
            'result': result,
            'text_length': len(cleaned_text),
            'fact_checks': fact_checks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time analysis"""
    await manager.connect(websocket)
    
    try:
        await manager.send_message({
            'type': 'connection',
            'message': 'Connected to Fake News Detector',
            'model_loaded': model is not None
        }, websocket)
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Send processing status
            await manager.send_message({
                'type': 'status',
                'message': 'Processing your request...'
            }, websocket)
            
            text = data.get('text', '')
            url = data.get('url', '')
            
            # Extract from URL if provided
            if url:
                await manager.send_message({
                    'type': 'status',
                    'message': 'Extracting content from URL...'
                }, websocket)
                url_text = extract_article_from_url(url)
                if url_text:
                    text = url_text
            
            # Clean text
            cleaned_text = clean_text(text)
            
            if not cleaned_text or len(cleaned_text) < 20:
                await manager.send_message({
                    'type': 'error',
                    'message': 'Text too short for analysis. Please provide more content.'
                }, websocket)
                continue
            
            # Predict
            await manager.send_message({
                'type': 'status',
                'message': 'Analyzing content...'
            }, websocket)
            
            result = predict_news(cleaned_text)
            
            # Get fact checks if fake
            fact_checks = []
            if result['prediction'] == 'FAKE':
                fact_checks = get_fact_check_info(cleaned_text[:100])
            
            # Send result
            await manager.send_message({
                'type': 'result',
                'data': {
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'method': result.get('method', 'unknown'),
                    'probabilities': {
                        'fake': result.get('fake_probability', 0),
                        'real': result.get('real_probability', 0)
                    },
                    'text_length': len(cleaned_text),
                    'fact_checks': fact_checks
                }
            }, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await manager.send_message({
                'type': 'error',
                'message': str(e)
            }, websocket)
        except:
            pass
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)
