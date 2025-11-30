from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load ML model & vectorizer
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("spam_model.pkl")

app = FastAPI()

class EmailData(BaseModel):
    subject: str
    body: str

def extract_top_keywords(text, top_n=5):
    """
    Extract suspicious keywords based on model weights.
    """
    # Transform email into vector
    X = vectorizer.transform([text])

    # Get model weights
    coef = model.coef_[0]

    # Multiply vector values with model weights
    scores = X.toarray()[0] * coef

    # Get top keyword indices
    top_indices = scores.argsort()[-top_n:][::-1]

    # Map indices → words
    feature_names = vectorizer.get_feature_names_out()
    keywords = [feature_names[i] for i in top_indices if X.toarray()[0][i] > 0]

    return keywords

@app.post("/analyze-email")
def analyze(email: EmailData):
    text = email.subject + " " + email.body

    # 1. Predict spam probability
    vector = vectorizer.transform([text])
    spam_prob = model.predict_proba(vector)[0][1]

    # 2. Extract suspicious keywords
    keywords = extract_top_keywords(text)

    response = {
        "spam_probability": round(float(spam_prob), 3),
        "result": "SPAM" if spam_prob > 0.5 else "SAFE",
        "suspicious_keywords": keywords
    }

    return response
