import os
import json
import pandas as pd
from django.shortcuts import render
import pickle

# Define base directory for dynamic file path handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def landing_page(request):
    return render(request, 'landing_page.html')

def visualizations_view(request):
    state_data = {
        'Maharashtra': 75,
        'Karnataka': 60,
        'Delhi': 85,
        'Gujarat': 50,
    }
    return render(request, 'visualizations.html', {'state_data': json.dumps(state_data)})

def all_visualizations(request):
    state_data = {
        "states": ["State1", "State2", "State3"],
        "percentages": [65, 75, 55],
    }
    modi_sentiment = {
        "labels": ["Positive", "Neutral", "Negative"],
        "values": [120, 60, 30],
    }
    modi_trends = {
        "dates": ["2024-01", "2024-02", "2024-03"],
        "scores": [0.5, 0.6, 0.4],
    }
    context = {
        "state_data": json.dumps(state_data),
        "modi_sentiment": json.dumps(modi_sentiment),
        "modi_trends": json.dumps(modi_trends),
    }
    return render(request, "all_visualizations.html", context)

def load_model():
    model_path = os.path.join(BASE_DIR, 'datasets', 'cleaned', 'winning_prediction_model.pkl')
    print(f"Loading model from: {model_path}")  # Debug

    import joblib  # Use joblib for loading
    model = joblib.load(model_path)
    return model


def preprocess_features(state_data):
    """
    Preprocess the input features to match the trained model's expectations.
    Assumes 'votes' needs conversion to 'vote_percentage'.
    """
    # Load the total votes dataset (hypothetical)
    total_votes_path = os.path.join(BASE_DIR, 'datasets', 'cleaned', 'processed_winner_data.csv')
    total_votes_df = pd.read_csv(total_votes_path)
    
    # Find the total votes for the matching state and year
    for index, row in state_data.iterrows():
        state, year, votes = row['state'], row['year'], row['votes']
        total_votes = total_votes_df[
            (total_votes_df['state'] == state) & (total_votes_df['year'] == year)
        ]['total_votes'].values
        
        if len(total_votes) > 0:
            vote_percentage = (votes / total_votes[0]) * 100
        else:
            print(f"Warning: No total votes data for state={state}, year={year}")
            vote_percentage = 0  # Fallback
        
        state_data.at[index, 'vote_percentage'] = vote_percentage

    # Drop the 'votes' column as the model expects 'vote_percentage'
    return state_data[['state', 'year', 'vote_percentage']]


def predict_winning_party(features):
    """
    Predict the winning party for given features.
    Handles both preprocessing and predictions.
    """
    model = load_model()
    processed_features = preprocess_features(features)
    print(f"Processed features for prediction: {processed_features}")  # Debug

    # Ensure the features match the model's input format
    prediction = model.predict(processed_features)
    print(f"Predicted winning party: {prediction}")  # Debug
    return prediction[0]


def winning_prediction(request):
    winning_party = None
    error_message = None

    if request.method == "POST":
        state = request.POST.get('state')
        year = request.POST.get('year')
        votes = request.POST.get('votes')

        print(f"Received input: state={state}, year={year}, votes={votes}")  # Debug

        if state and year and votes:
            try:
                # Create a DataFrame with user inputs
                state_data = pd.DataFrame({
                    'state': [state],
                    'year': [int(year)],
                    'votes': [int(votes)],
                })
                print(f"DataFrame for prediction: {state_data}")  # Debug

                # Predict the winning party
                winning_party = predict_winning_party(state_data)

            except Exception as e:
                error_message = f"Error during prediction: {str(e)}"
                print(error_message)  # Log error

    return render(
        request, 
        'winning_prediction.html', 
        {'winning_party': winning_party, 'error_message': error_message}
    )


from django.shortcuts import render
from ml_models.sentiment_analysis_model import analyze_sentiment

def sentiment_analysis_view(request):
    sentiment_result = None

    if request.method == "POST":
        tweet = request.POST.get('tweet')

        if tweet:
            # Use the preloaded analyze_sentiment function
            sentiment_result = analyze_sentiment(tweet)

    return render(request, 'sentiment_analysis.html', {'sentiment_result': sentiment_result})



from django.shortcuts import render
from django.http import JsonResponse
from visualizations.state_sentiment_visualization import visualize_winning_party

def state_winning_visualization(request):
    if request.method == "POST":
        state = request.POST.get("state")
        year = request.POST.get("year")

        if not state or not year:
            return JsonResponse({"error": "State and year are required."}, status=400)

        try:
            # Call the visualization function
            output_path = visualize_winning_party(state, int(year), output_dir="media/")
            if not output_path:
                return JsonResponse({"error": "No data found for the provided inputs."}, status=404)

            # Return the relative image path for Django to serve
            relative_path = output_path.replace("\\", "/").split("media/")[1]
            return JsonResponse({"image_path": f"media/{relative_path}"}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    # Render a form template if it's a GET request
    return render(request, "state_winning_visualization.html")
