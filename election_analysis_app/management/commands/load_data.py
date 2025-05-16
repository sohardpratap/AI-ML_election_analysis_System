from django.core.management.base import BaseCommand
import pandas as pd
from election_analysis_app.models import TweetSentiment
from datetime import datetime

class Command(BaseCommand):
    help = "Load tweet sentiment data into the database"

    def handle(self, *args, **kwargs):
        # Load the CSV
        file_path = "datasets/cleaned/cleaned_modi_tweets_with_sentiment.csv"
        df = pd.read_csv(file_path)

        # Convert 'Date' to strings and handle missing or invalid values
        def process_date(date):
            try:
                # Ensure date is in string format and valid
                return datetime.fromisoformat(str(date)).isoformat()
            except (ValueError, TypeError):
                return None  # Handle invalid or missing dates

        df['Date'] = df['Date'].apply(process_date)

        # Filter out rows with invalid dates
        df = df[df['Date'].notnull()]

        # Insert data into the database
        for _, row in df.iterrows():
            TweetSentiment.objects.create(
                tweet=row['Tweet'],
                sentiment=row['sentiment'],
                state=row['state'],
                date=row['Date']
            )

        self.stdout.write(self.style.SUCCESS("Data loaded successfully."))
