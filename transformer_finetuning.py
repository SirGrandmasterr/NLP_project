import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from preprocessor import TextPreprocessor

# --- Configuration ---
MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"
BATCH_SIZE = 16
NUM_EPOCHS = 1  # Kept low for demonstration; increase for better results
MAX_SEQ_LEN = 128
LEARNING_RATE = 2e-5

def load_data():
    """Loads and preprocesses data."""
    print("Loading data...")
    try:
        df = pd.read_csv('IMDB Dataset.csv')
        # Reducing size for speed during development/demo
        # df = df.sample(2000, random_state=42) 
        print(f"Loaded {len(df)} reviews.")
    except FileNotFoundError:
        print("IMDB Dataset.csv not found. Trying small dataset...")
        df = pd.read_csv('IMDB Dataset_small.csv')
        print(f"Loaded {len(df)} reviews from small dataset.")

    # Basic Preprocessing (HTML removal)
    # Transformers handle raw text well, but basic cleaning is good.
    preprocessor = TextPreprocessor(
        remove_html=True,
        lowercase=True, 
        remove_punctuation=False, # BERT needs punctuation
        remove_stopwords=False,  # BERT needs context
        lemmatize=False
    )
    
    print("Preprocessing text...")
    df['clean_review'] = df['review'].apply(preprocessor.process_text)
    
    # Map sentiment to labels
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

def compare_tokenization(texts):
    """
    Demonstrates two tokenization approaches: WordPiece (MiniLM) vs BPE (RoBERTa).
    """
    print("\n" + "="*50)
    print("TOKENIZATION COMPARISON: WordPiece vs. Byte-Pair Encoding")
    print("="*50)
    
    sample_text = texts[0]
    print(f"Sample Text (First 100 chars): {sample_text[:100]}...\n")

    # 1. WordPiece (used by MiniLM/BERT)
    wp_tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
    wp_tokens = wp_tokenizer.tokenize(sample_text)
    print(f"--- WordPiece (MiniLM) ---")
    print(f"Tokens: {wp_tokens[:15]}...")
    print(f"Total Tokens: {len(wp_tokens)}")
    
    # 2. Byte-Pair Encoding (used by RoBERTa/GPT)
    bpe_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    bpe_tokens = bpe_tokenizer.tokenize(sample_text)
    print(f"\n--- Byte-Pair Encoding (RoBERTa) ---")
    print(f"Tokens: {bpe_tokens[:15]}...")
    print(f"Total Tokens: {len(bpe_tokens)}")
    print("="*50 + "\n")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    
    return {
        'accuracy': acc,
        'macro_f1': f1,
    }

def main():
    # 1. Load Data
    df = load_data()
    
    # 2. Tokenization Comparison Requirement
    compare_tokenization(df['clean_review'].tolist())
    
    # 3. Split Data (Train/Validation/Test)
    # Split: 70% Train, 15% Val, 15% Test
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['clean_review'], df['label'], test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )
    
    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}, Test size: {len(test_texts)}")

    # 4. Tokenization for Training
    print("Tokenizing for training...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_SEQ_LEN)

    # Convert to Hugging Face Datasets
    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_ds = Dataset.from_dict({"text": test_texts, "label": test_labels})
    
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_val = val_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)

    # 5. Model Setup
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )
    
    # 7. Train
    print("Starting training...")
    trainer.train()
    
    # 8. Evaluation
    print("Evaluating on Test Set...")
    predictions = trainer.predict(tokenized_test)
    preds = np.argmax(predictions.predictions, axis=-1)
    
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average='macro')
    cm = confusion_matrix(test_labels, preds)
    
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("="*30)

if __name__ == "__main__":
    main()
