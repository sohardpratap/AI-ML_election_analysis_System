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
    # --- State-level Election Data ---
    state_election_chart_data = {"labels": [], "data": [], "error": None}
    try:
        election_data_path = os.path.join(BASE_DIR, 'datasets', 'cleaned', 'cleaned_big_election_dataset.csv')
        election_df = pd.read_csv(election_data_path)
        if not election_df.empty:
            # Example: Uttar Pradesh, 2019. Change if this specific data isn't insightful or available.
            # Check if 'Uttar Pradesh' and 2019 exist
            filtered_election_df = election_df[(election_df['state'] == 'Uttar Pradesh') & (election_df['year'] == 2019)]

            if filtered_election_df.empty: # Fallback to first available state/year if UP/2019 not found
                if not election_df.empty:
                    first_state = election_df['state'].iloc[0]
                    first_year = election_df['year'].iloc[0]
                    filtered_election_df = election_df[(election_df['state'] == first_state) & (election_df['year'] == first_year)]
                    state_election_chart_data["info"] = f"Displaying data for {first_state}, {first_year} (Uttar Pradesh, 2019 not found)."
                else:
                    state_election_chart_data["error"] = "Election data is empty."

            if not filtered_election_df.empty:
                state_election_chart_data['labels'] = filtered_election_df['candidate'].tolist()
                state_election_chart_data['data'] = filtered_election_df['votes'].tolist()
        else:
            state_election_chart_data["error"] = "Election data CSV is empty."
    except FileNotFoundError:
        state_election_chart_data["error"] = "Election data CSV not found."
    except Exception as e:
        state_election_chart_data["error"] = f"Error processing election data: {str(e)}"

    # --- Modi Sentiment Data ---
    modi_sentiment_chart_data = {"labels": [], "data": [], "error": None}
    modi_trend_chart_data = {"dates": [], "scores": [], "error": None}
    try:
        modi_tweets_path = os.path.join(BASE_DIR, 'datasets', 'cleaned', 'cleaned_modi_tweets_with_sentiment.csv')
        modi_tweets_df = pd.read_csv(modi_tweets_path)
        if not modi_tweets_df.empty and 'sentiment' in modi_tweets_df.columns:
            sentiment_counts = modi_tweets_df['sentiment'].value_counts()
            modi_sentiment_chart_data['labels'] = sentiment_counts.index.tolist()
            modi_sentiment_chart_data['data'] = sentiment_counts.values.tolist()

            # Sentiment Trends (Modi)
            if 'Timestamp' in modi_tweets_df.columns: # Assuming 'Timestamp' column
                try:
                    modi_tweets_df['datetime'] = pd.to_datetime(modi_tweets_df['Timestamp'], errors='coerce')
                    modi_tweets_df.dropna(subset=['datetime'], inplace=True) # Drop rows where conversion failed

                    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
                    modi_tweets_df['sentiment_score'] = modi_tweets_df['sentiment'].map(sentiment_map)

                    # Aggregate by month/year
                    modi_tweets_df['month_year'] = modi_tweets_df['datetime'].dt.to_period('M')
                    monthly_sentiment = modi_tweets_df.groupby('month_year')['sentiment_score'].mean().reset_index()
                    monthly_sentiment.sort_values('month_year', inplace=True) # Ensure chronological order

                    modi_trend_chart_data['dates'] = monthly_sentiment['month_year'].astype(str).tolist()
                    modi_trend_chart_data['scores'] = monthly_sentiment['sentiment_score'].tolist()
                except Exception as e:
                    modi_trend_chart_data["error"] = f"Error processing Modi sentiment trends: {str(e)}"
            else:
                modi_trend_chart_data["error"] = "Timestamp column not found for Modi sentiment trends."
        else:
            modi_sentiment_chart_data["error"] = "Modi sentiment data CSV is empty or 'sentiment' column missing."
            modi_trend_chart_data["error"] = "Modi sentiment data CSV is empty or 'sentiment' column missing."

    except FileNotFoundError:
        modi_sentiment_chart_data["error"] = "Modi sentiment data CSV not found."
        modi_trend_chart_data["error"] = "Modi sentiment data CSV not found."
    except Exception as e:
        modi_sentiment_chart_data["error"] = f"Error processing Modi sentiment data: {str(e)}"
        modi_trend_chart_data["error"] = f"Error processing Modi sentiment data: {str(e)}"

    # --- Rahul Sentiment Data ---
    rahul_sentiment_chart_data = {"labels": [], "data": [], "error": None}
    # As 'cleaned_rahul_tweets_with_sentiment.csv' was not found, we'll set an error message.
    rahul_sentiment_chart_data["error"] = "Rahul Gandhi sentiment data with pre-computed sentiments not found. Analysis skipped."
    # If 'cleaned_rahul_tweets_with_sentiment.csv' were available, the logic would be similar to Modi's sentiment:
    # try:
    #     rahul_tweets_path = os.path.join(BASE_DIR, 'datasets', 'cleaned', 'cleaned_rahul_tweets_with_sentiment.csv')
    #     rahul_tweets_df = pd.read_csv(rahul_tweets_path)
    #     if not rahul_tweets_df.empty and 'sentiment' in rahul_tweets_df.columns:
    #         sentiment_counts = rahul_tweets_df['sentiment'].value_counts()
    #         rahul_sentiment_chart_data['labels'] = sentiment_counts.index.tolist()
    #         rahul_sentiment_chart_data['data'] = sentiment_counts.values.tolist()
    #     else:
    #         rahul_sentiment_chart_data["error"] = "Rahul sentiment data CSV is empty or 'sentiment' column missing."
    # except FileNotFoundError:
    #     rahul_sentiment_chart_data["error"] = "Rahul sentiment data CSV not found."
    # except Exception as e:
    #     rahul_sentiment_chart_data["error"] = f"Error processing Rahul sentiment data: {str(e)}"

    context = {
        "state_election_chart_data": json.dumps(state_election_chart_data),
        "modi_sentiment_chart_data": json.dumps(modi_sentiment_chart_data),
        "modi_trend_chart_data": json.dumps(modi_trend_chart_data),
        "rahul_sentiment_chart_data": json.dumps(rahul_sentiment_chart_data),
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
            raise ValueError(f"Historical total votes data not found for state {state} and year {year}, cannot calculate vote percentage.")
        
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
    try:
        prediction = model.predict(processed_features)
    except ValueError as e:
        if "found unknown categories" in str(e).lower():
            raise ValueError("The model is not trained for the provided state/year combination.")
        else:
            raise e  # Re-raise other ValueErrors
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

            except ValueError as ve:
                error_str = str(ve)
                if "Historical total votes data not found" in error_str:
                    error_message = f"Prediction failed: Historical total votes data not found for the state '{state}' and year '{year}'. Please try a different combination."
                elif "The model is not trained for the provided state/year combination" in error_str:
                    error_message = f"Prediction failed: The model is not trained for the state '{state}' and year '{year}'. Please try a different combination."
                else:
                    error_message = f"Error during prediction: {error_str}"
                print(error_message) # Log error
            except Exception as e:
                error_message = f"An unexpected error occurred: {str(e)}"
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
from visualizations.winning_party_visualization import visualize_winning_party

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
