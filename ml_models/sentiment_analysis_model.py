
import pandas as pd
from transformers import pipeline

# Load the sentiment analysis pipeline once
sentiment_analyzer = pipeline('sentiment-analysis')

# Function to analyze sentiment of a single tweet
def analyze_sentiment(tweet):
    if isinstance(tweet, str) and tweet.strip():  # Only process non-empty strings
        result = sentiment_analyzer(tweet)[0]
        return result['label']
    return "NEUTRAL"  # Handle empty or invalid inputs

# Optional: Function to analyze and save sentiments for a dataset
def analyze_and_save_sentiments(input_path, output_path):
    # Load the cleaned dataset
    df = pd.read_csv(input_path)

    # Perform sentiment analysis on the 'cleaned_tweet' column
    if 'cleaned_tweet' in df.columns:
        df['sentiment'] = df['cleaned_tweet'].apply(analyze_sentiment)

        # Save the results back to a file
        df.to_csv(output_path, index=False)
        print(f"Sentiment analysis results saved to {output_path}")
    else:
        print("The 'cleaned_tweet' column is missing in the dataset.")
