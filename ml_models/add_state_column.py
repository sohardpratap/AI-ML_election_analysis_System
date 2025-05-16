import pandas as pd
import random

# Load the dataset
modi_df = pd.read_csv('datasets/cleaned/cleaned_modi_tweets_with_sentiment.csv')

# List of possible states (for demonstration purposes, you can expand this list)
states = ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "West Bengal", "Uttar Pradesh"]

# Add a random state to each row (or you can manually assign specific states if needed)
modi_df['state'] = [random.choice(states) for _ in range(len(modi_df))]

# Save the dataset with the added 'state' column
modi_df.to_csv('datasets/cleaned/cleaned_modi_tweets_with_sentiment.csv', index=False)

print("State column added successfully.")
