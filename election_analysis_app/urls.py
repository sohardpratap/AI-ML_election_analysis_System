from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing_page'),  # Landing page
    # path('visualizations/', views.visualizations_view, name='visualizations'),
    path("visualization/", views.state_winning_visualization, name="state_winning_visualization"),
    path('all_visualizations/', views.all_visualizations, name='all_visualizations'),
    path('sentiment_analysis/', views.sentiment_analysis_view, name='sentiment_analysis'),
    path('winning_prediction/', views.winning_prediction, name='winning_prediction'),
]
