# 🧠 Sentiment Analysis Web App (Streamlit)

This is a Streamlit-based web app for real-time **Sentiment Analysis** of user input and uploaded CSV files. It uses a trained machine learning model to classify sentiment into Positive, Negative, or Neutral.

## 🚀 Features

- Real-time sentiment analysis with emoji/color feedback
- Prediction history within session
- CSV file upload for bulk analysis
- Downloadable results
- Model retraining via uploaded dataset

## 📦 How to Run

### Locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 Train Your Own Model
Your dataset should have two columns:

- text: the input sentence
- sentiment: one of positive, neutral, or negative

You can upload this file directly in the app to retrain the model.