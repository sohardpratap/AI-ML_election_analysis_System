import pandas as pd
from transformers import pipeline

# Load the cleaned tweets
modi_df = pd.read_csv('datasets/cleaned/cleaned_modi_tweets.csv')

# Define the sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

# Analyze sentiment
def analyze_sentiment(tweet):
    if isinstance(tweet, str) and tweet.strip():  # Only process non-empty strings
        result = sentiment_analyzer(tweet)[0]
        return result['label']
    return "NEUTRAL"  # Handle empty or invalid inputs

modi_df['sentiment'] = modi_df['cleaned_tweet'].apply(analyze_sentiment)

# Save the results to a CSV file
output_path = 'datasets/cleaned/cleaned_modi_tweets_with_sentiment.csv'
modi_df.to_csv(output_path, index=False)

print(f"Sentiment analysis results saved to {output_path}")

