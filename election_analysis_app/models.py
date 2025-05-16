from django.db import models

class ElectionData(models.Model):
    state = models.CharField(max_length=100)
    year = models.IntegerField()
    candidate = models.CharField(max_length=100)
    party = models.CharField(max_length=100)
    votes = models.IntegerField()

    def __str__(self):
        return f"{self.state} - {self.year}"

class TweetSentiment(models.Model):
    tweet = models.TextField()
    sentiment = models.CharField(max_length=50)
    state = models.CharField(max_length=100)
    date = models.DateTimeField()

    def __str__(self):
        return f"Sentiment for tweet: {self.tweet[:50]}..."
