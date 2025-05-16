import pandas as pd
import os

# Define paths
CLEANED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/cleaned")
PROCESSED_FILE = os.path.join(CLEANED_DIR, "processed_winner_data.csv")
RAW_BIG_FILE = os.path.join(CLEANED_DIR, "cleaned_big_election_dataset.csv")

print("Adding 'total_votes' column...")

# Load datasets
processed_df = pd.read_csv(PROCESSED_FILE)
big_df = pd.read_csv(RAW_BIG_FILE)

# Ensure the big dataset has necessary columns
if 'state' in big_df.columns and 'year' in big_df.columns and 'votes' in big_df.columns:
    # Compute total_votes per state and year
    big_df['total_votes'] = big_df.groupby(['state', 'year'])['votes'].transform('sum')
    
    # Merge total_votes into the processed dataset
    merged_df = pd.merge(
        processed_df,
        big_df[['state', 'year', 'total_votes']].drop_duplicates(),
        on=['state', 'year'],
        how='left'
    )
    
    # Save the updated dataset
    merged_df.to_csv(PROCESSED_FILE, index=False)
    print("Successfully added 'total_votes' and saved updated dataset!")
else:
    print("The big dataset does not have the necessary columns (state, year, votes).")
