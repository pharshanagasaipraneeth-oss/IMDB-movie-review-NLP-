
import streamlit as st
import numpy as np
import re
import nltk
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec, KeyedVectors

# Download required NLTK resources
for resource in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

# Load trained ANN model
model2 = tf.keras.models.load_model("glove_ann_model.keras")

# Load saved GloVe embedding model
try:
    glove_model = Word2Vec.load("glove_embedding.model").wv
except Exception:
    glove_model = KeyedVectors.load("glove_embedding.model")

# Initialize preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return tokens


# Convert review into GloVe vector
def get_review_vector(tokens):
    vectors = [
        glove_model[word]
        for word in tokens
        if word in glove_model
    ]

    if len(vectors) == 0:
        return np.zeros(100, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


# Streamlit application
st.set_page_config(
    page_title="IMDB Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below to predict whether "
    "the sentiment is Positive or Negative."
)

review = st.text_area("Enter your movie review:", height=150)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:
        tokens = clean_text(review)

        review_vector = get_review_vector(tokens)

        input_data = np.array(
            [review_vector],
            dtype=np.float32
        )

        prediction = model2.predict(input_data, verbose=0)

        score = float(np.asarray(prediction).reshape(-1)[0])

        st.subheader("Prediction Result")

        if score >= 0.5:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")

        st.write(f"Prediction confidence: {score:.2%}")
        st.write("Preprocessed Review:")
        st.write(" ".join(tokens))
