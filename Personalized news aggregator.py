import streamlit as st
from newspaper import Article
from newspaper import build
from transformers import pipeline
import os
import requests
from requests.exceptions import RequestException
import tensorflow as tf
import re

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set TensorFlow threading options
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Load the summarizer pipeline
summarizer = pipeline("summarization")

# Streamlit app layout
st.title("Personalized News Aggregator")

# Function to clean text by removing unwanted characters
def clean_text(text):
    # Remove any unwanted characters or formatting
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = text.strip()  # Remove leading and trailing whitespace
    return text

# Function to fetch articles from a news source
@st.cache_data
def fetch_articles(source_url):
    try:
        response = requests.get(source_url, timeout=10)  # Set a timeout for the request
        response.raise_for_status()  # Raise an error for bad responses
        paper = build(source_url, memoize_articles=False)
    except RequestException as e:
        st.error(f"Error fetching articles from {source_url}: {e}")
        return []

    articles = []
    for article in paper.articles[:2]:  # Limit to 2 articles per source
        try:
            article.download()
            article.parse()
            cleaned_text = clean_text(article.text)  # Clean the article text
            if cleaned_text:  # Only add articles with valid text
                articles.append({
                    'title': article.title,
                    'text': cleaned_text,
                    'url': article.url
                })
        except Exception as e:
            st.error(f"Error processing article from {source_url}: {e}")
            continue  # Continue to the next article
    return articles

# Summarize the text using the Hugging Face pipeline
@st.cache_data
def summarize_text(text, max_length=80):  # Reduced max_length for shorter summaries
    # Chunking to avoid token limit issues
    chunks = [text[i:i + 512] for i in range(0, len(text), 512)]
    batch_summaries = summarizer(chunks, max_length=max_length, min_length=10, do_sample=False)
    summarized_text = " ".join([summary['summary_text'] for summary in batch_summaries if summary['summary_text']])
    return summarized_text or "Summary could not be generated."

# Function to display summarized news articles
def display_articles(articles):
    for article in articles:
        st.header(article['title'])
        st.markdown(f"[Read full article here]({article['url']})")  # Link to the full article
        summary = summarize_text(article['text'])
        st.write("**Summary:**")
        st.write(summary)
        st.write("---")  # Add a horizontal line for better separation between articles

# Sidebar for category selection
category = st.sidebar.selectbox("Choose News Category", ("Indian News", "Sports", "Technology", "World", "Entertainment", "Health", "Finance"))

# Button to refresh articles
if st.sidebar.button("Refresh"):
    st.caching.clear_cache()  # Clear cache when refreshing

# URLs for different categories
news_sources = {
    "Indian News": [
        "https://timesofindia.indiatimes.com",
        "https://www.thehindu.com",
        "https://www.ndtv.com"
    ],
    "Sports": [
        "https://www.espn.com",
        "https://www.sportskeeda.com",
        "https://www.cricbuzz.com"
    ],
    "Technology": [
        "https://techcrunch.com",
        "https://www.theverge.com",
        "https://www.wired.com"
    ],
    "World": [
        "https://www.bbc.com/news",
        "https://www.aljazeera.com",
        "https://www.cnn.com"
    ],
    "Entertainment": [
        "https://www.bollywoodhungama.com",
        "https://www.hollywoodreporter.com",
        "https://www.variety.com"
    ],
    "Health": [
        "https://www.webmd.com",
        "https://www.healthline.com",
        "https://www.medicalnewstoday.com"
    ],
    "Finance": [
        "https://www.moneycontrol.com",
        "https://www.financialexpress.com",
        "https://www.forbes.com/finance"
    ]
}

# Fetch and display news articles based on the selected category
if category in news_sources:
    for source in news_sources[category]:
        st.subheader(f"Fetching news from: {source}")
        articles = fetch_articles(source)
        if articles:
            display_articles(articles)
        else:
            st.warning(f"No articles found for {source}.")
