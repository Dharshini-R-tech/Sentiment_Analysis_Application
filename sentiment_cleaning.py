import re
import nltk
from nltk.corpus import stopwords

# Download stopwords once
nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    # Remove @mentions and hashtags
    text = re.sub(r'@\w+|#', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)