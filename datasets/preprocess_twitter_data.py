import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the dataset (replace with your actual path)
modi_df = pd.read_csv('datasets/raw/ModiRelatedTweetsWithSentiment.csv')

# Clean tweet function
def clean_tweet(text):
    # Ensure the input is a string (handle NaN or None)
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove mentions (e.g., @user)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (e.g., #hashtag)
    text = re.sub(r'#\w+', '', text)
    
    # Remove special characters, numbers, and punctuations
    text = re.sub(r'[^A-Za-z\s]', '', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text

# Apply the cleaning function to the 'Tweet' column
modi_df['cleaned_tweet'] = modi_df['Tweet'].apply(clean_tweet)

# Show the first few rows of the cleaned data
print(modi_df[['Tweet', 'cleaned_tweet']].head())

# Save the cleaned data to a new CSV (optional)
modi_df.to_csv('datasets/cleaned/cleaned_modi_tweets.csv', index=False)
