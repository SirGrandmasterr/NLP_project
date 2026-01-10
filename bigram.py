import math
import random
import time
import pandas as pd
from collections import defaultdict
from preprocessor import TextPreprocessor

class BigramLanguageModel:
    def __init__(self, alpha=0.01):
        """
        Initialize the Bigram Model.
        
        Args:
            alpha (float): The smoothing parameter for Laplace smoothing. 
                           Default is 0.01.
        """
        self.alpha = alpha
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)
        self.vocab = set()
        self.vocab_size = 0
        self.total_bigrams = 0
        self.total_unigrams = 0
        
        self.lambda1 = 0.3 # Unigram
        self.lambda2 = 0.7 # Bigram
        
    def train(self, corpus):
        """
        Trains the model on a corpus of tokenized sentences.
        Uses the full vocabulary (no <UNK> thresholding).
        """
        print("Training model on full vocabulary...")
        for sentence in corpus:
            # Update vocabulary and unigram counts
            for word in sentence:
                self.vocab.add(word)
                self.unigram_counts[word] += 1
                self.total_unigrams += 1
            
            # Update bigram counts
            for i in range(len(sentence) - 1):
                w_curr = sentence[i]
                w_next = sentence[i+1]
                self.bigram_counts[w_curr][w_next] += 1
                self.total_bigrams += 1
                
        self.vocab_size = len(self.vocab)
        print(f"Training complete. Vocab size: {self.vocab_size}")

    def get_probability(self, prev_word, word):
        """
        Calculates the interpolated probability P(word | prev_word).
        P = L2 * P(word|prev) + L1 * P(word)
        """
        # 1. Bigram Probability
        bigram_count = self.bigram_counts[prev_word][word]
        unigram_count_prev = self.unigram_counts[prev_word]
        
        p_bi_num = bigram_count + self.alpha
        p_bi_den = unigram_count_prev + (self.alpha * self.vocab_size)
        p_bi = p_bi_num / p_bi_den
        
        # 2. Unigram Probability
        unigram_count_word = self.unigram_counts[word]
        p_uni_num = unigram_count_word + self.alpha
        p_uni_den = self.total_unigrams + (self.alpha * self.vocab_size)
        p_uni = p_uni_num / p_uni_den
        
        return (self.lambda2 * p_bi) + (self.lambda1 * p_uni)

    def calculate_perplexity(self, test_corpus):
        """
        Calculates the perplexity of the model on a test corpus.
        """
        log_prob_sum = 0
        N = 0
        
        for sentence in test_corpus:
            for i in range(len(sentence) - 1):
                w_curr = sentence[i]
                w_next = sentence[i+1]
                
                # We do not replace with <UNK>. If a word is unknown,
                # get_probability handles it via smoothing.
                prob = self.get_probability(w_curr, w_next)
                
                log_prob_sum += math.log2(prob)
                N += 1
        
        if N == 0: return float('inf')
        
        avg_log_prob = -log_prob_sum / N
        perplexity = 2 ** avg_log_prob
        return perplexity

    def generate_sentence(self, max_length=20):
        """
        Generates a random sentence.
        """
        current_word = "<s>"
        sentence = [current_word]
        
        for _ in range(max_length):
            if current_word == "</s>":
                break
                
            # If current_word was never seen in training (e.g. from a user prompt),
            # unigram_count is 0. We fallback to uniform distribution or break.
            # Here we sample from the whole vocab if unknown, or just observed followers if known.
            
            possible_next = self.bigram_counts[current_word]
            
            if not possible_next:
                # Dead end or unknown word. 
                # Ideally: Sample uniformly from V (or weighted by unigrams).
                # For efficiency/simplicity here: break or pick random.
                break 

            candidates = list(possible_next.keys())
            counts = list(possible_next.values())
            
            next_word = random.choices(candidates, weights=counts, k=1)[0]
            
            sentence.append(next_word)
            current_word = next_word
            
        return " ".join(sentence)

    def autocomplete(self, prompt, preprocessor, max_length=20):
        """
        Completes a given text prompt.
        """
        cleaned_prompt = preprocessor.process_text(prompt)
        tokens = cleaned_prompt.split()
        
        if not tokens:
            current_word = "<s>"
        else:
            current_word = tokens[-1]
            
        # Warning: If current_word is not in self.vocab, generation will stop immediately
        # because bigram_counts[current_word] will be empty.
        
        generated_tokens = []
        for _ in range(max_length):
            if current_word == "</s>":
                break
            
            possible_next = self.bigram_counts[current_word]
            
            if not possible_next:
                break
                
            candidates = list(possible_next.keys())
            counts = list(possible_next.values())
            
            next_word = random.choices(candidates, weights=counts, k=1)[0]
            
            generated_tokens.append(next_word)
            current_word = next_word
            
        return prompt + " " + " ".join(generated_tokens)

    def autocomplete(self, prompt, preprocessor, max_length=20):
        """
        Completes a given text prompt using the trained model.
        """
        # Preprocess the prompt to get the last token
        cleaned_prompt = preprocessor.process_text(prompt)
        tokens = cleaned_prompt.split()
        
        if not tokens:
            current_word = "<s>"
        else:
            current_word = tokens[-1]
            
        # Handle OOV for the seed word
        if current_word not in self.vocab:
            # Optionally print a warning or fallback
            current_word = "<UNK>"
            
        # Generate continuation
        generated_tokens = []
        for _ in range(max_length):
            if current_word == "</s>":
                break
            
            possible_next = self.bigram_counts[current_word]
            
            if not possible_next:
                # If we hit a dead end (should be rare with smoothing context, but possible if UNK), replace
                current_word = "<UNK>"
                possible_next = self.bigram_counts[current_word]

            if not possible_next:
                break
                
            candidates = list(possible_next.keys())
            counts = list(possible_next.values())
            
            next_word = random.choices(candidates, weights=counts, k=1)[0]
            
            generated_tokens.append(next_word)
            current_word = next_word
            
        return prompt + " " + " ".join(generated_tokens)

# --- Example Usage with IMDB Data ---

def dummy_preprocessor(text):
    """
    Placeholder for your existing pipeline.
    Ensures <s> and </s> are added and text is tokenized.
    """
    # Simple tokenization for demonstration
    tokens = text.lower().strip().split()
    return ['<s>'] + tokens + ['</s>']

def main():
    # 1. Load Data
    try:
        # Assuming the CSV is in the same directory
        # Using the column names from your screenshot: 'review', 'sentiment'
        df = pd.read_csv('IMDB Dataset.csv')
        print("Dataset loaded successfully.")
        
        # Taking a subset for demonstration speed
        reviews = df['review'].tolist() 
        
    except FileNotFoundError:
        print("IMDB Dataset.csv not found. Using dummy data.")
        reviews = ["The movie was terrible.", "I loved the movie."]

    # 2. Apply Preprocessing (Less aggressive for Language Modeling)
    preprocessor = TextPreprocessor(
        remove_html=True,
        lowercase=True,
        remove_punctuation=False, # Keep punctuation for structure
        remove_stopwords=False,   # Keep stopwords for grammar
        lemmatize=True,          # Keep original word forms
        expand_contractions=True)
        
    tokenized_corpus = []
    print("Preprocessing texts...")
    for r in reviews:
        # Preprocessor returns a single string of space-separated tokens
        cleaned_text = preprocessor.process_text(r)
        # Split into list of tokens
        tokens = cleaned_text.split()
        # Add sentence boundaries
        tokens = ['<s>'] + tokens + ['</s>']
        tokenized_corpus.append(tokens)

    # 3. Split Train/Test
    split_idx = int(len(tokenized_corpus) * 0.8)
    train_data = tokenized_corpus[:split_idx]
    test_data = tokenized_corpus[split_idx:]

    # 4. Initialize and Train
    model = BigramLanguageModel(alpha=0.01) # Reduced alpha
    
    start_time = time.time()
    model.train(train_data)
    end_time = time.time()
    print(f"Time to build model: {end_time - start_time:.4f} seconds")

    # 5. Generate Text
    print("\n--- Generated Reviews ---")
    for _ in range(3):
        print(f"- {model.generate_sentence()}")

    # 6. Evaluate Perplexity
    print("\n--- Evaluation ---")
    pp = model.calculate_perplexity(test_data)
    print(f"Model Perplexity on Test Set: {pp:.2f}")
    
    # 7. Autocomplete Demo
    print("\n--- Autocomplete Demo ---")
    prompts = [
        "The movie was",
        "I really liked",
        "The acting is",
        "This film is a complete"
    ]
    for p in prompts:
        completed = model.autocomplete(p, preprocessor)
        print(f"Prompt: '{p}'\nResult: {completed}\n")

if __name__ == "__main__":
    main()