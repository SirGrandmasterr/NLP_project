import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import pandas as pd
import numpy as np
import random
from collections import Counter
from preprocessor import TextPreprocessor

# --- Configuration ---
EMBED_DIM = 64
HIDDEN_DIM = 256 # Increased for full dataset
BATCH_SIZE = 64 # Increased for full dataset
LEARNING_RATE = 0.005
NUM_EPOCHS = 6 # Reduced for full dataset because it's much larger
MAX_SEQ_LEN = 100  # Limit max length to avoid OOM or super long padding

class Vocabulary:
    def __init__(self, token_to_idx=None):
        if token_to_idx:
            self.token_to_idx = token_to_idx
        else:
            self.token_to_idx = {"<PAD>": 0, "<UNK>": 1, "<s>": 2, "</s>": 3}
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        
    def build_vocab(self, sentences, min_freq=2):
        print("Building vocabulary...")
        all_tokens = [token for sent in sentences for token in sent]
        counts = Counter(all_tokens)
        
        for token, count in counts.items():
            if count >= min_freq and token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.token_to_idx)
                
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        print(f"Vocabulary size: {len(self.token_to_idx)}")
        
    def __len__(self):
        return len(self.token_to_idx)
    
    def stoi(self, token):
        return self.token_to_idx.get(token, self.token_to_idx["<UNK>"])
        
    def itos(self, idx):
        return self.idx_to_token.get(idx, "<UNK>")

class IMDBDataset(Dataset):
    def __init__(self, sentences, vocab):
        self.sentences = sentences
        self.vocab = vocab
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        tokenized_sent = self.sentences[idx]
        # Numericalize
        indexed = [self.vocab.stoi(t) for t in tokenized_sent]
        return torch.tensor(indexed, dtype=torch.long)

def collate_fn(batch):
    """
    Custom collate function to handle variable length sentences via padding.
    """
    # batch is a list of tensors
    # Sort by length (descending) for pack_padded_sequence
    batch.sort(key=lambda x: len(x), reverse=True)
    
    # Separate source and target
    # Source: <s> w1 w2 ... wn
    # Target: w1 w2 ... wn </s>
    # Actually, our sentences in 'sentences' list usually have <s> and </s> already.
    # So we just take :-1 as input and 1: as target.
    
    inputs = [item[:-1] for item in batch]
    targets = [item[1:] for item in batch]
    
    lengths = torch.tensor([len(x) for x in inputs], dtype=torch.long)
    
    # Pad sequences
    # padding_value=0 is <PAD>
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return inputs_padded, targets_padded, lengths

class NeuralLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(NeuralLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, lengths=None, hidden=None):
        # x: (batch, seq_len)
        embed = self.embedding(x) # (batch, seq_len, embed_dim)
        
        if lengths is not None:
            # Pack
            packed_embed = pack_padded_sequence(embed, lengths.cpu(), batch_first=True, enforce_sorted=True)
            packed_out, hidden = self.lstm(packed_embed, hidden)
            # Unpack
            output, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            # No packing (e.g. inference)
            output, hidden = self.lstm(embed, hidden)
            
        # output: (batch, seq_len, hidden_dim) (padded where needed)
        
        logits = self.fc(output) # (batch, seq_len, vocab_size)
        return logits, hidden

def generate_text(model, vocab, start_prompt="The movie", max_len=20, device='cpu', temperature=1.0):
    model.eval()
    preprocessor = TextPreprocessor(lowercase=True)
    tokens = preprocessor.process_text(start_prompt).split()
    
    current_idx = [vocab.stoi(t) for t in tokens]
    # Add start token if not present logic? 
    # The model trained on <s>... so prompt should ideally start with something logical.
    # If we feed "The movie", it's mid-sentence-ish.
    
    input_seq = torch.tensor(current_idx, dtype=torch.long).unsqueeze(0).to(device) # (1, seq_len)
    
    generated = list(tokens)
    
    hidden = None
    
    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model(input_seq, hidden=hidden)
            
            # Get last time step
            last_logits = logits[0, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                last_logits = last_logits / temperature
                
            probs = torch.softmax(last_logits, dim=0)
            
            # Sample
            next_token_idx = torch.multinomial(probs, 1).item()
            next_token = vocab.itos(next_token_idx)
            
            if next_token == "</s>":
                break
                
            generated.append(next_token)
            
            # Next input is the single token we just generated (feeding back one by one)
            # Or we could feed the whole sequence, but feeding 1 is efficient IF we keep hidden state.
            input_seq = torch.tensor([[next_token_idx]], dtype=torch.long).to(device)
            
    return " ".join(generated)

def main():
    # Check for device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load and Preprocess Data
    try:
        df = pd.read_csv('IMDB Dataset.csv')
        reviews = df['review'].tolist()
        print(f"Loaded {len(reviews)} reviews from full dataset.")
        # Optional: Slice to speeding up if user wants, e.g. reviews = reviews[:10000]
    except FileNotFoundError:
        print("IMDB Dataset.csv not found. Checking for small dataset...")
        try:
             df = pd.read_csv('IMDB Dataset_small.csv')
             reviews = df['review'].tolist()
             print(f"Loaded {len(reviews)} reviews from small dataset.")
        except FileNotFoundError:
             print("No dataset found.")
             return

    preprocessor = TextPreprocessor(
        remove_html=True,
        lowercase=True,
        expand_contractions=True,
        remove_punctuation=False # Embedding models handle punctuation well
    )
    
    print("Preprocessing...")
    tokenized_sentences = []
    for r in reviews:
        # Simple split by preprocessor
        txt = preprocessor.process_text(r)
        toks = txt.split()
        # Add boundaries
        toks = ['<s>'] + toks[:MAX_SEQ_LEN] + ['</s>']
        tokenized_sentences.append(toks)
        
    # 2. Build Vocabulary
    vocab = Vocabulary()
    vocab.build_vocab(tokenized_sentences, min_freq=2)
    
    # 3. Prepare Dataset/Loader
    dataset = IMDBDataset(tokenized_sentences, vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # 4. Initialize Model
    model = NeuralLM(len(vocab), EMBED_DIM, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore <PAD>
    
    # 5. Training Loop
    print("Starting training...")
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for inputs, targets, lengths in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs shape: (B, L), targets: (B, L)
            
            optimizer.zero_grad()
            
            # Forward
            logits, _ = model(inputs, lengths) 
            # logits: (B, L, V)
            
            # Flatten for loss
            # Reshape logits to (B*L, V), targets to (B*L)
            loss = criterion(logits.view(-1, len(vocab)), targets.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")
        
    # 6. Generation Demo
    print("\n--- Generated Text ---")
    prompts = ["The movie was", "I think", "This is"]
    for p in prompts:
        gen = generate_text(model, vocab, p, device=device, temperature=0.8) # Slightly lowered temp for coherence
        print(f"Prompt: '{p}' -> {gen}")

if __name__ == "__main__":
    main()
