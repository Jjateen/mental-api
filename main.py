from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
from pydantic import BaseModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}}) 

# Initialize SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Load models
model_save_dir = 'P:/BHUHackathon/mentalApi/mental-api/models'
models = [joblib.load(f'{model_save_dir}/model{i}.pkl') for i in range(9)]  # Assuming you have 9 models

# Pydantic model for input validation
class PostText(BaseModel):
    post_text: str

# Function to preprocess text
def preprocess_text(text):
    # Remove hyperlinks
    text = re.sub(r'http\S+', '', text)
    return text

# Function to calculate sentiment scores
def calculate_sentence(text):
    scores = sid.polarity_scores(text)
    return scores['compound'], scores['neg'], scores['neu']

# Function to predict using the models
def predict(post_text):
    # Preprocess text
    preprocessed_text = preprocess_text(post_text)
    
    # Calculate sentiment scores
    sentiment_scores = calculate_sentence(preprocessed_text)
    
    # Prepare data for prediction
    data = [sentiment_scores[0], sentiment_scores[1], sentiment_scores[2]]
    
    # Make predictions using each model
    predictions = {}
    class_labels = [
        "Feeling-bad-about-yourself-or-that-you-are-a-failure-or-have-let-yourself-or-your-family-down",
        "Feeling-down-depressed-or-hopeless",
        "Feeling-tired-or-having-little-energy",
        "Little-interest-or-pleasure-in-doing",
        "Moving-or-speaking-so-slowly-that-other-people-could-have-noticed-Or-the-opposite-being-so-fidgety-or-restless-that-you-have-been-moving-around-a-lot-more-than-usual",
        "Poor-appetite-or-overeating",
        "Thoughts-that-you-would-be-better-off-dead-or-of-hurting-yourself-in-some-way",
        "Trouble-concentrating-on-things-such-as-reading-the-newspaper-or-watching-television",
        "Trouble-falling-or-staying-asleep-or-sleeping-too-much"
    ]
    
    for i, model in enumerate(models):
        class_name = class_labels[i]
        predictions[class_name] = int(model.predict([data])[0])
    
    return predictions

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Parse and validate input using Pydantic
        data = PostText.parse_obj(request.get_json())
        post_text = data.post_text
        
        # Make predictions
        predictions = predict(post_text)
        
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)