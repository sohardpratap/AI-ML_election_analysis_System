import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with sentiments
data_path = 'E:/ai_election_analysis/election_analysis/datasets/cleaned/cleaned_modi_tweets_with_sentiment.csv'
df = pd.read_csv(data_path)

# Visualization 1: Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='sentiment', palette='viridis')
plt.title('Sentiment Distribution in Modi-related Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('e:\\ai_election_analysis\\election_analysis\\visualizations\\modi_sentiment_distribution.png')
plt.show()

# Visualization 2: Sentiment Trends Over Time
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # Remove rows with invalid dates

    sentiment_trends = df.groupby([df['Date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
    sentiment_trends.plot(figsize=(10, 6), title="Sentiment Trends Over Time")
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.savefig('e:\\ai_election_analysis\\election_analysis\\visualizations\\modi_sentiment_trends.png')
    plt.show()
else:
    print("No 'Date' column found. Skipping time-based sentiment trends.")
