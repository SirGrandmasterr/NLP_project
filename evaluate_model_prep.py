import pandas as pd
import numpy as np
import torch
import requests
import json
import os
import sys
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from preprocessor import TextPreprocessor

# --- Configuration ---
MODEL_PATH = "./results"  # Path to the saved model (checkpoint)
# Try to find the latest checkpoint if explicit path not found, or just use the base output dir if it contains the model
DATASET_PATH = "IMDB Dataset.csv"
SMALL_DATASET_PATH = "IMDB Dataset_small.csv"
OUTPUT_CSV = "evaluation_set.csv"
LLM_API_URL = "http://localhost:1234/v1/chat/completions"
LLM_MODEL_NAME = "local-model" # Usually ignored by local servers like LM Studio, but good to have
NUM_SAMPLES = 100

def load_data():
    """Loads and preprocesses data, mirroring transformer_finetuning.py"""
    print("Loading data...")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"{DATASET_PATH} not found. Trying small dataset...")
        try:
            df = pd.read_csv(SMALL_DATASET_PATH)
        except FileNotFoundError:
            print("No dataset found.")
            sys.exit(1)

    # Basic Preprocessing
    preprocessor = TextPreprocessor(
        remove_html=True,
        lowercase=True, 
        remove_punctuation=False,
        remove_stopwords=False,
        lemmatize=False
    )
    
    print("Preprocessing text...")
    df['clean_review'] = df['review'].apply(preprocessor.process_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

def get_test_set(df):
    """Reproduces the test split from transformer_finetuning.py"""
    # Split: 70% Train, 15% Val, 15% Test
    # Random state 42 is CRITICAL to get the same split
    _, temp_texts, _, temp_labels = train_test_split(
        df['clean_review'], df['label'], test_size=0.3, random_state=42
    )
    _, test_texts, _, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )
    
    test_df = pd.DataFrame({'text': test_texts, 'gold_label': test_labels})
    return test_df

def get_llm_judgment(text):
    """Queries the local LLM to judge the sentiment."""
    headers = {"Content-Type": "application/json"}
    
    # Prompt engineering for the judge
    system_prompt = "You are an expert sentiment analyst. Analyze the following movie review and determine if it is Positive or Negative. Return ONLY one word: 'positive' or 'negative'."
    user_prompt = f"Review: \"{text}\"\n\nSentiment:"
    
    data = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0, # Deterministic
        "max_tokens": 10
    }
    
    try:
        response = requests.post(LLM_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content'].strip().lower()
        
        if "positive" in content:
            return 1
        elif "negative" in content:
            return 0
        else:
            print(f"Warning: Ambiguous LLM response: {content}")
            return -1 # Error/Unknown
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to LLM server. Is LM Studio running on port 1234?")
        return -2
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return -2

def main():
    # 1. Prepare Data
    df_full = load_data()
    test_df = get_test_set(df_full)
    
    # Subsample 100
    if len(test_df) < NUM_SAMPLES:
        print(f"Warning: Test set smaller than {NUM_SAMPLES}, using full test set.")
        subsample = test_df
    else:
        subsample = test_df.sample(n=NUM_SAMPLES, random_state=999) # New random state for subsampling
    
    print(f"Subsampled {len(subsample)} instances.")

    # 2. Model Predictions
    # We need to load the trained model. 
    # If ./results/checkpoint-X exists, we might need to point there, 
    # but AutoModelForSequenceClassification often handles local dirs if they contain config/model.bin
    
    print("Loading finetuned model...")
    # Attempt to find best checkpoint if base dir not valid
    model_to_load = MODEL_PATH
    if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        # Check for checkpoints
        checkpoints = [d for d in os.listdir(MODEL_PATH) if d.startswith("checkpoint")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[1])) # Sort by step
            model_to_load = os.path.join(MODEL_PATH, checkpoints[-1]) # Take latest
            print(f"Found checkpoint: {model_to_load}")
        else:
            print(f"Warning: No clean model found in {MODEL_PATH}. Using 'microsoft/MiniLM-L12-H384-uncased' (base) as fallback for demo.")
            model_to_load = "microsoft/MiniLM-L12-H384-uncased"
            
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_to_load)
        model = AutoModelForSequenceClassification.from_pretrained(model_to_load)
    except Exception as e:
        print(f"Failed to load model from {model_to_load}: {e}")
        return

    # Helper for prediction
    def predict_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        return pred_id

    print("Running Model Predictions...")
    subsample['model_prediction'] = subsample['text'].apply(predict_sentiment)
    
    # 3. LLM Predictions
    print("Running LLM-as-a-judge (this may take a while)...")
    subsample['llm_label'] = subsample['text'].apply(get_llm_judgment)
    
    # 4. Prepare Evaluation Columns
    subsample['human_label_1'] = ""
    subsample['human_label_2'] = ""
    
    # Reorder columns
    cols = ['text', 'gold_label', 'model_prediction', 'llm_label', 'human_label_1', 'human_label_2']
    final_df = subsample[cols]
    
    # Save
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully saved evaluation set to {OUTPUT_CSV}")
    print("ACTION REQUIRED: Please open this CSV and fill in 'human_label_1' and 'human_label_2' columns.")

if __name__ == "__main__":
    main()
