import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to cleaned dataset
DATASET_PATH = os.path.join("datasets", "cleaned", "cleaned_big_election_dataset.csv")

# DATASET_PATH = os.path.join(os.path.dirname(__file__), '../../datasets/cleaned/cleaned_big_election_dataset.csv')

def visualize_winning_party(state: str, year: int, output_dir="../media/"):
    """
    Visualize the winning party data for a given state and year.
    
    Args:
        state (str): Name of the state.
        year (int): Year of the election.
        output_dir (str): Directory to save the visualization.
        
    Returns:
        str: Path to the saved visualization image.
    """
    # Load the dataset
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Dataset not found at {DATASET_PATH}")
        return None
    
    # Filter the data for the given state and year
    filtered_df = df[(df['state'] == state) & (df['year'] == year)]
    
    if filtered_df.empty:
        print(f"No data found for state: {state} and year: {year}")
        return None
    
    # Find the winning party
    winning_party = filtered_df.loc[filtered_df['votes'].idxmax()]
    winning_party_name = winning_party['party']
    winning_candidate = winning_party['candidate']
    
    # Plot the votes for all candidates
    plt.figure(figsize=(10, 6))
    plt.bar(filtered_df['candidate'], filtered_df['votes'], color='skyblue')
    plt.title(f"Winning Party in {state}, {year}: {winning_party_name} ({winning_candidate})")
    plt.xlabel("Candidates")
    plt.ylabel("Votes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the visualization
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"winning_party_{state}_{year}.png")
    plt.savefig(output_path)
    plt.close()
    
    print(f"Visualization saved at {output_path}")
    return output_path

# Example Usage
if __name__ == "__main__":
    visualize_winning_party(state="Uttar Pradesh", year=2019)
