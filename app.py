import os
import re
import pickle
import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)

# --- MODEL AND VECTOR LOADING ---
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# --- SIMPLE FAST TEXT CLEANING ---
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z ]+', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens) if tokens else text

# --- SQLITE DB SETUP ---
DB_FILE = 'reviews.db'
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            product_name TEXT,
            rating INTEGER,
            review_text TEXT,
            sentiment TEXT
        )
    ''')
    conn.commit()
    conn.close()
init_db()

# --- PREDICTION ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    review = data.get('review', '')
    review_clean = clean_text(review)
    review_vec = vectorizer.transform([review_clean]).toarray()
    pred = model.predict(review_vec)[0]
    response = {0: "Sorry for that", 1: "Thanks for such sweet review", 2: "Thank you"}
    return jsonify({"result": response.get(pred, "Thank you")})

# --- SAVE REVIEW ENDPOINT ---
@app.route('/save-review', methods=['POST'])
def save_review():
    data = request.get_json()
    review_text = data['reviewText']
    review_clean = clean_text(review_text)
    review_vec = vectorizer.transform([review_clean]).toarray()
    pred = model.predict(review_vec)[0]
    sentiment = {0: "bad", 1: "good", 2: "neutral"}.get(pred, "neutral")
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO reviews (name, email, product_name, rating, review_text, sentiment)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (data['name'], data['email'], data['productName'], data['rating'], data['reviewText'], sentiment))
    conn.commit()
    conn.close()
    return
