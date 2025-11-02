import os
import re
import pickle
import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)

# --- NLTK DATA CHECK ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# --- MODEL AND VECTOR LOADING ---
try:
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except Exception as e:
    import traceback, sys
    print("Error loading model/vectorizer:", e, file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)
    exit(1)

# --- SIMPLE FAST TEXT CLEANING ---
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
    return jsonify({"status": "success", "sentiment": sentiment})

# --- GET ALL REVIEWS ENDPOINT ---
@app.route('/get-reviews', methods=['GET'])
def get_reviews():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM reviews')
    rows = c.fetchall()
    conn.close()
    reviews = [{
        "id": r[0], "name": r[1], "email": r[2], "product_name": r[3], "rating": r[4],
        "review_text": r[5], "sentiment": r[6]
    } for r in rows]
    return jsonify(reviews)

# --- ERROR HANDLER FOR DEBUGGING ---
@app.errorhandler(Exception)
def handle_exception(e):
    import traceback, sys
    print("ERROR:", e, file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)
    return jsonify({"error": "Internal server error"}), 500

# --- RUN SERVER ON RENDER ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
