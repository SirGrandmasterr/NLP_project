# Neural Language Model Explanation

This document explains the implementation of the Neural Language Model found in `neural_lm.py`.

## 1. Model Architecture

The model consists of three main layers (`NeuralLM` class):

1.  **Embedding Layer**: 
    - Converts integer token indices into dense vectors of fixed size (`EMBED_DIM=64`).
    - Handles `<PAD>` tokens by assigning them a zero vector (via `padding_idx=0`).
    
2.  **LSTM Layer**:
    - A Long Short-Term Memory (LSTM) network processes the sequence of embeddings.
    - It maintains an internal hidden state that captures the context of the sentence seen so far.
    - We use `pack_padded_sequence` and `pad_packed_sequence`. This allows the LSTM to efficiently process variable-length sentences in a batch without computing on the padding tokens.

3.  **Linear (Output) Layer**:
    - Projects the LSTM output (`hidden_dim=128`) back to the vocabulary size.
    - The output logits represent the unnormalized probability scores for the next word.

## 2. Data Preparation

### Preprocessing
We use the same `TextPreprocessor` as the N-gram models. We add `<s>` and `</s>` tokens to mark sentence boundaries.

### Vocabulary
We build a vocabulary from the training data, filtering out rare words (`min_freq=2`).
- Special tokens:
    - `<PAD>` (0): Padding for batching.
    - `<UNK>` (1): Unknown words.
    - `<s>` (2): Start of sentence.
    - `</s>` (3): End of sentence.

### Batching & Padding
To train on complete sentences of different lengths simultaneously, we need padding:
- **`collate_fn`**: This function is called by the DataLoader.
    - It sorts the batch by sentence length (required for `pack_padded_sequence`).
    - It separates inputs (first word to second-to-last) and targets (second word to last).
    - It pads the sequences with `0` so they all match the length of the longest sentence in the batch.

## 3. Training Process

- **Loss Function**: `CrossEntropyLoss` is used. Crucially, we set `ignore_index=0`, which tells the loss function to completely ignore errors on `<PAD>` tokens. This prevents the model from "learning" to just predict padding.
- **Optimizer**: We use `Adam` with a learning rate of 0.005.
- **Loop**: For each batch, we predict the next word for every position in the sequence and update weights to minimize the prediction error.

## 4. Text Generation

The `generate_text` function demonstrates the model's capability:
1.  Takes a prompt text.
2.  Feeds it into the model to get the initial hidden state.
3.  Samples the next word from the probability distribution (Softmax of logits).
4.  Feeds the generated word back into the model to generate the next one.
5.  Repeats until `</s>` or max length is reached.
