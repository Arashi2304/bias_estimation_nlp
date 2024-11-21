import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import json
import os

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def load_file(file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            return file.readlines()
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as file:
            return json.load(file).keys()  # Use the keys as prompts if JSON contains key-value pairs
    else:
        return []

def analyze_sentiment(text):
    result = nlp(text)
    return result

# List of files to process
input_files = [
    'gender_task1_results.json',
    'sexuality_task1_results.json',
    'ethnicity_task1_results.json'
]

output_dir = 'sentiment_results'
os.makedirs(output_dir, exist_ok=True)

for input_file in input_files:
    print(f"Processing {input_file}...")
    texts = load_file(input_file)

    sentiment_results = []
    for text in texts:
        sentiment = analyze_sentiment(text)
        sentiment_results.append({
            'text': text,
            'sentiment': sentiment[0]['label'],
            'score': sentiment[0]['score']
        })

    output_file = os.path.join(output_dir, f"{os.path.basename(input_file).split('.')[0]}_sentiment.json")
    with open(output_file, 'w') as result_file:
        json.dump(sentiment_results, result_file, indent=4)

    print(f"Sentiment results saved to {output_file}.")
