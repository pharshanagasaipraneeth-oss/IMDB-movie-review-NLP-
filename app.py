
import streamlit as st
import numpy as np
import re
import nltk
import tensorflow as tf

from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors


# --------------------------------
# 1. Streamlit Page Configuration
# --------------------------------

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


# --------------------------------
# 2. Define Project Directory
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------
# 3. Download Required NLTK Resources
# --------------------------------

@st.cache_resource
def download_nltk_resources():

    resources = [
        "stopwords",
        "punkt",
        "punkt_tab",
        "wordnet",
        "omw-1.4"
    ]

    for resource in resources:
        nltk.download(resource, quiet=True)


download_nltk_resources()


# --------------------------------
# 4. Load Trained ANN Model
# --------------------------------

@st.cache_resource
def load_ann_model():

    model_path = BASE_DIR / "glove_ann_model.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            f"ANN model not found: {model_path.name}"
        )

    return tf.keras.models.load_model(str(model_path))


# --------------------------------
# 5. Load GloVe Embedding Vectors
# --------------------------------

@st.cache_resource
def load_glove_vectors():

    glove_path = BASE_DIR / "glove_vectors.kv"

    if not glove_path.exists():
        raise FileNotFoundError(
            f"GloVe vectors not found: {glove_path.name}"
        )

    return KeyedVectors.load(str(glove_path))


# --------------------------------
# 6. Initialize Models and NLP Tools
# --------------------------------

try:

    model2 = load_ann_model()

    glove_model = load_glove_vectors()

    stop_words = set(stopwords.words("english"))

    lemmatizer = WordNetLemmatizer()

except Exception as e:

    st.error(f"Error loading project files: {e}")

    st.info(
        "Check that glove_ann_model.keras and "
        "glove_vectors.kv are present in your project folder."
    )

    st.stop()


# --------------------------------
# 7. Text Preprocessing
# --------------------------------

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



# --------------------------------
# 8. Convert Review into GloVe Vector
# --------------------------------

def get_review_vector(tokens):

    vectors = []

    for word in tokens:

        word = word.lower()

        try:
            # Check using GloVe vocabulary
            if word in glove_model.key_to_index:

                vector = glove_model.get_vector(word)

                vectors.append(vector)

        except KeyError:
            continue

    # Handle reviews with no matching words
    if len(vectors) == 0:

        return np.zeros(
            glove_model.vector_size,
            dtype=np.float32
        )

    # Calculate average word embedding
    review_vector = np.mean(
        vectors,
        axis=0
    ).astype(np.float32)

    return review_vector

# 9. Movie Review Input
# --------------------------------

review = st.text_area(
    "Enter your movie review:",
    height=150,
    placeholder="Type your movie review here..."
)


# --------------------------------
# 10. Sentiment Prediction
# --------------------------------

if st.button("Predict Sentiment"):

    if review.strip() == "":

        st.warning("Please enter a movie review.")

    else:

        try:

            # Preprocess review
            tokens = clean_text(review)

            # Convert review into GloVe vector
            review_vector = get_review_vector(tokens)

            # Check input dimensions
            expected_features = model2.input_shape[-1]

            if len(review_vector) != expected_features:

                st.error(
                    f"Input dimension mismatch. "
                    f"Model expects {expected_features} features, "
                    f"but GloVe produced {len(review_vector)}."
                )

                st.stop()

            # Prepare ANN input
            input_data = np.array(
                [review_vector],
                dtype=np.float32
            )

            # Predict sentiment
            prediction = model2.predict(
                input_data,
                verbose=0
            )

            score = float(
                np.asarray(prediction).reshape(-1)[0]
            )

            # Display result
            st.subheader("Prediction Result")

            if score >= 0.5:

                st.success("Positive Review 😊")

            else:

                st.error("Negative Review 😞")

            st.write(
                f"Model output score: {score:.2%}"
            )

            st.write("Preprocessed Review:")

            st.write(" ".join(tokens))

        except Exception as e:

            st.error(f"Prediction error: {e}")
