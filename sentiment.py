from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize the HuggingFace BERT model for sentiment analysis
bert_analyzer = pipeline('sentiment-analysis')

def analyze_with_vader(text):
    sentiment_score = analyzer.polarity_scores(text)
    # Determine the sentiment
    if sentiment_score['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_score['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, sentiment_score['compound']

def analyze_with_bert(text):
    result = bert_analyzer(text)[0]
    sentiment = result['label']
    score = result['score']
    return sentiment, score
