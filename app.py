import streamlit as st
import joblib
import pandas as pd
from sentiment_cleaning import clean_text

# Load the pretrained sentiment model
model = joblib.load('sentiment_model.pkl')

st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🧠 Sentiment Analysis with ML")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Emoji and color maps for sentiment results
sentiment_emojis = {
    "positive": "😊",
    "negative": "😠",
    "neutral": "😐"
}
sentiment_colors = {
    "positive": "green",
    "negative": "red",
    "neutral": "gray"
}

st.write("🔎 Enter a sentence to analyze its sentiment.")

# Text input for single sentence
user_input = st.text_area("Enter your sentence here:", height=100)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        prediction = model.predict([cleaned])[0]
        emoji = sentiment_emojis.get(prediction, "")
        color = sentiment_colors.get(prediction, "black")
        st.markdown(f"<h3 style='color:{color};'>Predicted Sentiment: {prediction.capitalize()} {emoji}</h3>", unsafe_allow_html=True)
        st.session_state.history.append((user_input, prediction))

# Show prediction history if checkbox is checked
if st.checkbox("📜 Show Prediction History"):
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history, columns=["Text", "Predicted Sentiment"])
        st.dataframe(df_hist)
    else:
        st.info("No predictions made yet.")

# Bulk CSV upload and prediction
st.write("📂 Upload a CSV file to analyze multiple sentences.")
uploaded_file = st.file_uploader("Upload CSV (must have 'text' column)", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            df['cleaned_text'] = df['text'].apply(clean_text)
            df['Predicted Sentiment'] = model.predict(df['cleaned_text'])
            df['Emoji'] = df['Predicted Sentiment'].map(sentiment_emojis)
            st.success("Predictions complete!")
            st.dataframe(df[['text', 'Predicted Sentiment', 'Emoji']])
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Results", csv, "sentiment_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")

# 📊 Sentiment Visualization Section
import matplotlib.pyplot as plt

st.header("📈 Sentiment Visualization")

uploaded_file = st.file_uploader("Upload a CSV file with a 'sentiment' column", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'sentiment' not in df.columns:
        st.error("CSV must contain a 'sentiment' column.")
    else:
        sentiment_counts = df['sentiment'].value_counts()

        # Pie chart
        st.subheader("Pie Chart of Sentiments")
        fig1, ax1 = plt.subplots()
        ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        # Bar chart
        st.subheader("Bar Chart of Sentiment Counts")
        fig2, ax2 = plt.subplots()
        sentiment_counts.plot(kind='bar', color=[
            'green' if s == 'positive' else 'red' if s == 'negative' else 'gray'
            for s in sentiment_counts.index
        ], ax=ax2)
        ax2.set_xlabel("Sentiment")
        ax2.set_ylabel("Count")
        ax2.set_title("Sentiment Distribution")
        st.pyplot(fig2)
