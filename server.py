from flask import Flask, jsonify, request
from flask_cors import CORS
from nltk.corpus import stopwords
app = Flask(__name__)
CORS(app)
@app.route('/', methods=['GET'])
def home():
    return jsonify("Hello world!")

@app.route('/fakenews', methods=['POST'])
def data_route():
import pandas as pd
import numpy as  np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
nltk.download('stopwords')
from nltk.corpus import stopwords
english_stopwords = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in english_stopwords])
    return text
# Load the dataset with the python engine and skip bad lines
fake_news = pd.read_csv('/content/Fake.csv', engine='python', on_bad_lines='skip')
real_news = pd.read_csv('/content/True.csv', engine='python', on_bad_lines='skip')
fake_news['label'] = 1
real_news['label'] = 0
news = pd.concat([fake_news, real_news], ignore_index=True)
news.drop_duplicates(subset='text', inplace=True)
news.dropna(subset=['text'], inplace=True)
news['text'] = news['text'].apply(preprocess_text)
print(news.info())
print(news.head())
print(news.isnull().sum())
news.info()
news.describe()
print("Dataset Shape:", news.shape)
print("Number of Fake News:", news[news['label'] == 1].shape[0])
print("Number of Real News:", news[news['label'] == 0].shape[0])
data_subset = news.sample(frac=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(data_subset['text'], data_subset['label'], test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
joblib.dump(pipeline, 'fake_news_detector_pipeline_rf.pkl')
print('Model saved as fake_news_detector_pipeline_rf.pkl')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
