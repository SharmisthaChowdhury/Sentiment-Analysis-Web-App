import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv('Combined Data.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df.dropna(subset=['statement'], inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['statement'], df['status'], test_size=0.2, random_state=42)

# Vectorize text
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train model
model = SVC()
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
joblib.dump(model, 'svc_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully.")
