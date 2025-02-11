import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords
nltk.download('stopwords')

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing function
portstem = PorterStemmer()
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    text = text.split()
    text = [portstem.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("💬 X (Twitter) Text Sentiment Analyzer")
st.subheader("Analyze the sentiment of tweets in real-time!")

# User Input
user_input = st.text_area("Enter a tweet:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        processed_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]

        # Display Result
        sentiment_label = "😊 Positive" if prediction == 1 else "😡 Negative"
        st.success(f"Predicted Sentiment: {sentiment_label}")
    else:
        st.warning("Please enter a tweet for analysis.")
