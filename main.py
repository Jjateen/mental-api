from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import json
import re
from nltk.corpus import stopwords

app = FastAPI()

# Allow requests from specified origins
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScoringItem(BaseModel):
    post_text: str

# Load sentiment analyzer
sid = SentimentIntensityAnalyzer()

target_class_names = [
    'Feeling-bad-about-yourself-or-that-you-are-a-failure-or-have-let-yourself-or-your-family-down',
    'Feeling-down-depressed-or-hopeless',
    'Feeling-tired-or-having-little-energy',
    'Little-interest-or-pleasure-in-doing',
    'Moving-or-speaking-so-slowly-that-other-people-could-have-noticed-Or-the-opposite-being-so-fidgety-or-restless-that-you-have-been-moving-around-a-lot-more-than-usual',
    'Poor-appetite-or-overeating',
    'Thoughts-that-you-would-be-better-off-dead-or-of-hurting-yourself-in-some-way',
    'Trouble-concentrating-on-things-such-as-reading-the-newspaper-or-watching-television',
    'Trouble-falling-or-staying-asleep-or-sleeping-too-much'
]
# Load models for each target
models = []
for i in range(9):  # Assuming 9 targets
    with open(f'model{i}.pkl', 'rb') as f:
        models.append(pickle.load(f))

# Define stopwords
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    # Remove hyperlinks
    text = re.sub(r'http\S+', '', text)
    # Remove stopwords
    text_tokens = text.split()
    filtered_text = [word for word in text_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    # Preprocess the input data
    item.post_text = preprocess_text(item.post_text)

    # Calculate sentiment scores
    text_compound, text_neg, text_neu = calculate_sentence(item.post_text)

    # Create a DataFrame for the input data
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    
    # Add sentiment scores to the DataFrame
    df['text_compound_score'] = text_compound
    df['text_negative_score'] = text_neg
    df['text_neutral_score'] = text_neu

    # Remove unnecessary columns
    df = df.drop(['post_title', 'post_text'], axis=1)

    # Make predictions for each model
    predictions = {}
    for i, model in enumerate(models):
        prediction = model.predict(df)
        target_name = target_class_names[i]
        predictions[target_name] = int(prediction[0])

    # Convert predictions to JSON string
    json_predictions = json.dumps(predictions)

    return json_predictions

def calculate_sentence(text):
    scores = sid.polarity_scores(text)
    return scores['compound'], scores['neg'], scores['neu']

@app.get('/')
async def read_root():
    return "Welcome to Mental Health Analyzer!"