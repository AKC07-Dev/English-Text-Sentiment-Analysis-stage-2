from flask import Flask, request, jsonify
import pickle
from googletrans import Translator
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)

# Load model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

translator = Translator()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z ]+', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens) if tokens else text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data['review']
    detected = translator.detect(review)
    if detected.lang != 'en':
        review = translator.translate(review, dest='en').text
    review_clean = clean_text(review)
    review_vec = vectorizer.transform([review_clean]).toarray()
    pred = model.predict(review_vec)[0]
    response = {0: "Sorry for that", 1: "Thanks for such sweet review", 2: "Thank you"}
    return jsonify({"result": response.get(pred, "Thank you")})

if __name__ == '__main__':
    app.run()
