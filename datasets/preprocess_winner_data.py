import pandas as pd
import os

# Define the path to the cleaned dataset
CLEANED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned")

# Load the cleaned big election dataset
df = pd.read_csv(os.path.join(CLEANED_DIR, "cleaned_big_election_dataset.csv"))

# Show the first few rows to understand the data
print(df.head())

# Calculate the total votes per state and year
df['total_votes'] = df.groupby(['state', 'year'])['votes'].transform('sum')

# Calculate the percentage of votes for each candidate in each state
df['vote_percentage'] = (df['votes'] / df['total_votes']) * 100

# Find the winning candidate/party in each state for each election year
df_winner = df.loc[df.groupby(['state', 'year'])['vote_percentage'].idxmax()]

# The target variable will be the 'party' of the winning candidate
df_winner['winner'] = df_winner['party']

# Drop the columns that won't be used for the model
df_winner = df_winner[['state', 'year', 'winner', 'vote_percentage']]

# Display the processed data
print(df_winner.head())

# Save this cleaned data for the model
df_winner.to_csv(os.path.join(CLEANED_DIR, "processed_winner_data.csv"), index=False)
