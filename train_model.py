import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle

# Load dataset (you’ll download this next!)
df = pd.read_csv("IMDB Dataset.csv")
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Train model
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['review'])
model = LinearSVC()
model.fit(X, df['sentiment'])

# Save model files
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
print("Model trained and saved!")