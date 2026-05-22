import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# Clean data
df = df.dropna(subset=["review", "sentiment"])

df["sentiment"] = df["sentiment"].astype(str).str.lower()

df = df[df["sentiment"].isin(["positive", "negative"])]

# Features and labels
X = df["review"]
y = df["sentiment"].map({"positive": 1, "negative": 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Streamlit UI
st.title("Customer Feedback Sentiment Analysis")

review = st.text_area("Enter Your Review")

if st.button("Predict"):

    review_vec = vectorizer.transform([review])

    prediction = model.predict(review_vec)

    if prediction[0] == 1:
        st.success("Positive Review 😊")
    else:
        st.error("Negative Review 😔")

