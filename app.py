from flask import Flask, render_template, request
import joblib
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Load model, vectorizer, and categories
model = joblib.load('model/classifier.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')
categories = ['positive', 'negative']  # The correct mapping for prediction labels


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return " ".join(filtered)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['text']
        clean_text = preprocess(input_text)
        vector = vectorizer.transform([clean_text])
        prediction = model.predict(vector)[0]  # Prediction returns a string ('positive' or 'negative')

        # Directly use the prediction as the category
        category = prediction

        return render_template('index.html', result=category, input_text=input_text)
    return render_template('index.html', result=None)


if __name__ == '__main__':
    app.run(debug=True)
