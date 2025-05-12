from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Example training data
X = ["I love programming", "Python is great", "I hate bugs"]
y = ["positive", "positive", "negative"]

# Convert text to numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train the model
model = MultinomialNB()
model.fit(X_vectorized, y)

# Save the trained model, vectorizer, and categories
joblib.dump(model, 'model/classifier.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

# Save the categories as a list
categories = ['positive', 'negative']  # This should match the labels in your training data
joblib.dump(categories, 'model/categories.pkl')
