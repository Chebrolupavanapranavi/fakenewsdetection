from flask import Flask, jsonify, request
from flask_cors import CORS
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def home():
    return jsonify("Hello world!")

# @app.route('/fakenews', methods=['POST'])
# def data_route():
#     nltk.download('stopwords')
#     english_stopwords = set(stopwords.words('english'))
#     def preprocess_text(text):
#         text = text.lower()
#         text = ' '.join([word for word in text.split() if word not in english_stopwords])
#         return text
#     fake_news = pd.read_csv('/content/Fake.csv', engine='python', on_bad_lines='skip')
#     real_news = pd.read_csv('/content/True.csv', engine='python', on_bad_lines='skip')
#     fake_news['label'] = 1
#     real_news['label'] = 0
#     data = pd.concat([fake_news, real_news], ignore_index=True)
#     data.drop_duplicates(subset='text', inplace=True)
#     data.dropna(subset=['text'], inplace=True)
#     data['text'] = data['text'].apply(preprocess_text)
#     data_subset = data.sample(frac=0.1, random_state=42)
#     X_train, X_test, y_train, y_test = train_test_split(data_subset['text'], data_subset['label'], test_size=0.2, random_state=42)
#     pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
#     ('rf', RandomForestClassifier(n_estimators=100, random_state=42))])
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f'Accuracy: {accuracy}')
#     report = classification_report(y_test, y_pred)
#     print(f'Classification Report:\n{report}')
#     cm = confusion_matrix(y_test, y_pred)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     plt.show()
#     cv_scores = cross_val_score(pipeline, data_subset['text'], data_subset['label'], cv=5)
#     print(f'Cross-validation scores: {cv_scores}')
#     print(f'Average cross-validation score: {np.mean(cv_scores)}')
#     train_sizes, train_scores, test_scores = learning_curve(
#         pipeline, data_subset['text'], data_subset['label'], cv=5, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.figure()
#     plt.title("Learning Curve (Random Forest)")
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     plt.grid()
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1, color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
#     plt.legend(loc="best")
#     plt.show()

if __name__ == '__main__':
    app.run(debug=True, port=5000)