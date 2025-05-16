# election_analysis_app/tests.py
from django.test import TestCase
from django.urls import reverse

class DashboardTests(TestCase):

    def test_all_visualizations_page(self):
        """Test that the all_visualizations page loads correctly"""
        response = self.client.get(reverse('all_visualizations'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'State-wise Winning Percentages')  # Check for content in the page

    def test_sentiment_analysis_page(self):
        """Test that the sentiment_analysis page loads correctly"""
        response = self.client.get(reverse('sentiment_analysis'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Modi Sentiment Distribution')  # Check for content in the page
