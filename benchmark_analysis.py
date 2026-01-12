import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

INPUT_CSV = "evaluation_set.csv"

def load_and_clean_data():
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found. Please run evaluate_model_prep.py first.")
        return None

    # Check if human labels are filled
    # We treat empty strings or NaNs as missing
    if df['human_label_1'].isnull().all() and df['human_label_2'].isnull().all():
        print("WARNING: Human label columns appear empty. Running analysis on available data only.")
        
    # Convert labels to numeric where possible, handle errors
    for col in ['gold_label', 'model_prediction', 'llm_label', 'human_label_1', 'human_label_2']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

def print_metrics(y_true, y_pred, name_true="Gold", name_pred="Model"):
    # Filter out NaNs (e.g. if LLM failed or human didn't annotate)
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]
    
    if len(y_t) == 0:
        print(f"No valid data to compare {name_true} vs {name_pred}")
        return

    acc = accuracy_score(y_t, y_p)
    f1 = f1_score(y_t, y_p, average='binary') # Assuming 0/1 
    
    print(f"--- {name_pred} vs {name_true} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(confusion_matrix(y_t, y_p))
    print("")

def main():
    print("Loading evaluation set...")
    df = load_and_clean_data()
    if df is None: return

    print(f"Loaded {len(df)} rows.")
    
    # 1. Metric-based Evaluation (Model vs Gold)
    print("\n=== 1. Metric-based Evaluation (Model vs Gold) ===")
    print_metrics(df['gold_label'], df['model_prediction'], "Gold", "Model")
    
    # 2. LLM Evaluation
    print("=== 2. LLM-as-a-judge Evaluation ===")
    print_metrics(df['gold_label'], df['llm_label'], "Gold", "LLM")
    print_metrics(df['model_prediction'], df['llm_label'], "Model", "LLM")
    
    # 3. Human Evaluation
    print("=== 3. Human Evaluation ===")
    
    # Aggregate human labels if both exist (e.g., take majority or mean rounded)
    # If they disagree, we can check Kappa
    
    h1 = df['human_label_1']
    h2 = df['human_label_2']
    
    if h1.isnull().all():
        print("No human annotations found. Skipping human evaluation metrics.")
    else:
        # Check Inter-Annotator Agreement if 2 annotators
        if not h2.isnull().all():
            # Kappa
            valid_k = ~np.isnan(h1) & ~np.isnan(h2)
            if valid_k.sum() > 0:
                kappa = cohen_kappa_score(h1[valid_k], h2[valid_k])
                print(f"Inter-Annotator Agreement (Cohen's Kappa): {kappa:.4f}")
            
            # Create a consensus label (if split, maybe trust Gold or random? For now, if disagree, treat as NaN or take H1)
            # Simple strategy: Average and round
            df['human_consensus'] = ((h1.fillna(h1) + h2.fillna(h1)) / 2).round()
        else:
            df['human_consensus'] = h1
            
        print_metrics(df['gold_label'], df['human_consensus'], "Gold", "Human")
        print_metrics(df['model_prediction'], df['human_consensus'], "Model", "Human")

if __name__ == "__main__":
    main()
