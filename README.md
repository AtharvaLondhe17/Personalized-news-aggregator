The Personalized News Aggregator is a Streamlit application that fetches, processes, and summarizes news articles from various online sources, delivering them in an easy-to-read format. It utilizes NLP techniques to automatically generate summaries of news content, enhancing user experience by condensing long articles into concise highlights.

Features
Fetches and displays articles from various news sources.
Summarizes article content using a transformer-based model (Hugging Face’s summarization pipeline).
Allows users to select from different news categories, such as Indian News, Sports, Technology, etc.
Limits the number of articles per source to improve processing efficiency.
Provides links to the full articles for further reading.
Cleanly formatted output with summaries to ensure readability.
Requirements
To run the project, the following dependencies are required:

Python 3.7+
Streamlit
Newspaper3k (for fetching and parsing news articles)
Hugging Face transformers library (for summarization)
TensorFlow (for performance optimization)
Requests (for handling HTTP requests)
