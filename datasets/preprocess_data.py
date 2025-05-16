import pandas as pd
import os

# Define paths for raw and cleaned datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of preprocess_data.py
RAW_DIR = os.path.join(BASE_DIR, "raw")
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")

# Ensure cleaned directory exists
os.makedirs(CLEANED_DIR, exist_ok=True)

# Function to preprocess the big election dataset
def preprocess_big_election_dataset(file_name):
    file_path = os.path.join(RAW_DIR, file_name)  # Correct path to the raw dataset
    # Load dataset
    df = pd.read_csv(file_path)
    print(f"Initial shape of {file_name}: {df.shape}")

    # Drop duplicates and handle missing values
    df = df.drop_duplicates().dropna()
    print(f"After cleaning {file_name}: {df.shape}")

    return df

# Function to preprocess tweets data
def preprocess_tweets(file_name):
    file_path = os.path.join(RAW_DIR, file_name)  # Correct path to the raw dataset
    # Load dataset
    df = pd.read_csv(file_path)
    print(f"Initial shape of {file_name}: {df.shape}")

    # Drop duplicates and handle missing values
    df = df.drop_duplicates().dropna()

    # Basic text cleaning
    df['Tweet'] = df['Tweet'].str.replace(r"http\S+", "", regex=True)  # Remove URLs
    df['Tweet'] = df['Tweet'].str.replace(r"[^a-zA-Z\s]", "", regex=True)  # Remove special chars
    df['Tweet'] = df['Tweet'].str.lower()  # Convert to lowercase

    print(f"After cleaning {file_name}: {df.shape}")
    return df

# Main execution
if __name__ == "__main__":
    # Process and save the cleaned datasets
    big_election_data = preprocess_big_election_dataset("big_election_dataset.csv")
    big_election_data.to_csv(os.path.join(CLEANED_DIR, "cleaned_big_election_dataset.csv"), index=False)

    tweets_data = preprocess_tweets("IndianElection19TwitterData.csv")
    tweets_data.to_csv(os.path.join(CLEANED_DIR, "cleaned_tweets.csv"), index=False)

    # Clean the additional datasets
    moditweets_data = preprocess_tweets("ModiRelatedTweetsWithSentiment.csv")
    moditweets_data.to_csv(os.path.join(CLEANED_DIR, "cleaned_modi_tweets.csv"), index=False)

    rahultweets_data = preprocess_tweets("RahulRelatedTweetsWithSentiment.csv")
    rahultweets_data.to_csv(os.path.join(CLEANED_DIR, "cleaned_rahul_tweets.csv"), index=False)

    tweetsmodi_data = preprocess_tweets("tweetsModi.csv")
    tweetsmodi_data.to_csv(os.path.join(CLEANED_DIR, "cleaned_tweets_modi.csv"), index=False)

    tweetsrg_data = preprocess_tweets("tweetsRG.csv")
    tweetsrg_data.to_csv(os.path.join(CLEANED_DIR, "cleaned_tweets_rg.csv"), index=False)



