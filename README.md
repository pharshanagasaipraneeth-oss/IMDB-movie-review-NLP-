# IMDB Movie Review Sentiment Analysis using NLP

## Project Overview

The **IMDB Movie Review Sentiment Analysis** project uses Natural Language Processing (NLP) and Deep Learning techniques to classify movie reviews into two categories: Positive and Negative.

The project uses GloVe word embeddings to convert textual reviews into numerical vectors and an Artificial Neural Network (ANN) to predict the sentiment.

A Streamlit web application is developed to allow users to enter movie reviews and receive sentiment predictions.

## Objectives

- Analyze movie reviews using NLP techniques.
- Perform text preprocessing and cleaning.
- Convert text into numerical representations using word embeddings.
- Build and evaluate deep learning models.
- Predict positive and negative movie sentiments.
- Deploy the trained model using Streamlit.

## Technologies Used

- Python
- Google Colab
- Jupyter Notebook
- Pandas
- NumPy
- NLTK
- Gensim
- TensorFlow / Keras
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

## Project Workflow

1. Dataset Collection
2. Exploratory Data Analysis (EDA)
3. Text Cleaning and Preprocessing
4. Tokenization
5. Stopword Removal
6. Lemmatization
7. Word Embedding using GloVe
8. Artificial Neural Network (ANN) Model Building
9. Model Evaluation
10. Model Saving
11. Streamlit Deployment

## Models and Techniques

The project explores different word embedding and deep learning approaches:

- CBOW (Continuous Bag of Words)
- Skip-Gram
- GloVe (Global Vectors for Word Representation)
- Artificial Neural Network (ANN)
- Simple Recurrent Neural Network (SimpleRNN)

The GloVe + ANN model was selected as the best-performing model in the experiments.

## Model Performance

| Model | Test Accuracy |
|---|---:|
| CBOW + ANN | 51.11% |
| Skip-Gram + ANN | 52.02% |
| GloVe + ANN | 80.90% |
| SimpleRNN | 50.19% |

*Note: These are the experimental accuracy values from the project.*

## Streamlit Web Application

The Streamlit application allows users to:

- Enter a movie review.
- Click the Predict Sentiment button.
- View the predicted sentiment.
- See the prediction confidence.

### Application Output

The model predicts one of the following:

- Positive 😊
- Negative 😞

## Project Structure

```text
IMDB-Movie-Review-Sentiment-Analysis/
│
├── app.py
├── requirements.txt
├── glove_ann_model.keras
├── glove_vectors.kv
├── NLP_project(IMDB_movie_review).ipynb
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd IMDB-Movie-Review-Sentiment-Analysis
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Sample Reviews

### Positive Review

```text
This movie was absolutely fantastic! The acting was brilliant,
the story was interesting, and I enjoyed every minute of it.
```

Expected Sentiment: Positive

### Negative Review

```text
This movie was terrible and extremely boring.
The acting was disappointing and the story was weak.
```

Expected Sentiment: Negative

## Future Improvements

- Experiment with LSTM and Bi-LSTM architectures.
- Improve model accuracy through hyperparameter tuning.
- Explore advanced word embeddings.
- Deploy the application online.
- Improve prediction performance on different types of movie reviews.

## Conclusion

This project demonstrates how Natural Language Processing and Deep Learning can be used to understand textual data and perform sentiment classification.

The trained GloVe + ANN model is integrated with a Streamlit application to provide an interactive movie review sentiment prediction system.

## Author

**Harsha**

Project: IMDB Movie Review Sentiment Analysis using NLP and Deep Learning
