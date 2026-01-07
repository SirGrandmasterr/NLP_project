import math
import random
import time
import pandas as pd
from collections import defaultdict
from preprocessor import TextPreprocessor

class TrigramLanguageModel:
    def __init__(self, alpha=0.01):
        """
        Initialize the Trigram Model.
        
        Args:
            alpha (float): The smoothing parameter for Laplace smoothing. 
                           Default is 0.01.
        """
        self.alpha = alpha
        # trigram_counts: count of (w1, w2, w3) aka given w1, w2, what is w3?
        # Structure: dict[(w1, w2)] -> dict[w3] -> count
        self.trigram_counts = defaultdict(lambda: defaultdict(int))
        
        # bigram_counts: count of (w1, w2) as a history.
        # Structure: dict[(w1, w2)] -> count
        self.bigram_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        
        self.vocab = set()
        self.vocab_size = 0
        self.total_trigrams = 0
        self.total_unigrams = 0
        
        # Interpolation weights
        self.lambda1 = 0.1 # Unigram
        self.lambda2 = 0.3 # Bigram
        self.lambda3 = 0.6 # Trigram
        
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
            
            # Update bigram counts (for backoff)
            for i in range(len(sentence) - 1):
                self.bigram_counts[(sentence[i], sentence[i+1])] += 1

            # Update trigram counts
            # Sentence is expected to be padded like ['<s>', '<s>', 'w1', ..., 'wn', '</s>']
            for i in range(len(sentence) - 2):
                w_1 = sentence[i]
                w_2 = sentence[i+1]
                w_3 = sentence[i+2]
                
                self.trigram_counts[(w_1, w_2)][w_3] += 1
                self.total_trigrams += 1
                
        self.vocab_size = len(self.vocab)
        print(f"Training complete. Vocab size: {self.vocab_size}")

    def get_probability(self, w_1, w_2, w_3):
        """
        Calculates the interpolated probability P(w_3 | w_1, w_2).
        P = L3 * P(w3|w1,w2) + L2 * P(w3|w2) + L1 * P(w3)
        """
        # 1. Trigram Probability
        trigram_count = self.trigram_counts[(w_1, w_2)][w_3]
        bigram_context_count = self.bigram_counts[(w_1, w_2)]
        
        p_tri_num = trigram_count + self.alpha
        p_tri_den = bigram_context_count + (self.alpha * self.vocab_size)
        p_tri = p_tri_num / p_tri_den
        
        # 2. Bigram Probability (Backoff)
        bigram_count = self.bigram_counts[(w_2, w_3)]
        unigram_context_count = self.unigram_counts[w_2]
        
        p_bi_num = bigram_count + self.alpha
        p_bi_den = unigram_context_count + (self.alpha * self.vocab_size)
        p_bi = p_bi_num / p_bi_den
        
        # 3. Unigram Probability
        unigram_count = self.unigram_counts[w_3]
        p_uni_num = unigram_count + self.alpha
        p_uni_den = self.total_unigrams + (self.alpha * self.vocab_size)
        p_uni = p_uni_num / p_uni_den
        
        return (self.lambda3 * p_tri) + (self.lambda2 * p_bi) + (self.lambda1 * p_uni)

    def calculate_perplexity(self, test_corpus):
        """
        Calculates the perplexity of the model on a test corpus.
        """
        log_prob_sum = 0
        N = 0
        
        for sentence in test_corpus:
            for i in range(len(sentence) - 2):
                w_1 = sentence[i]
                w_2 = sentence[i+1]
                w_3 = sentence[i+2]
                
                prob = self.get_probability(w_1, w_2, w_3)
                
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
        # Start with two padding tokens
        current_w1 = "<s>"
        current_w2 = "<s>"
        sentence = [current_w1, current_w2]
        
        for _ in range(max_length):
            # If we generated the end token, stop
            if current_w2 == "</s>":
                break
                
            possible_next = self.trigram_counts[(current_w1, current_w2)]
            
            if not possible_next:
                # If unknown history, we can't progress. 
                break 

            candidates = list(possible_next.keys())
            counts = list(possible_next.values())
            
            next_word = random.choices(candidates, weights=counts, k=1)[0]
            
            sentence.append(next_word)
            current_w1 = current_w2
            current_w2 = next_word
            
        # Return joined sentence, removing start tokens
        # Typically we don't show <s> <s>
        # The list has ['<s>', '<s>', 'word1', ... '</s>' maybe]
        # We can strip the first two <s>
        return " ".join(sentence[2:])

    def autocomplete(self, prompt, preprocessor, max_length=20):
        """
        Completes a given text prompt using the trained model.
        """
        cleaned_prompt = preprocessor.process_text(prompt)
        tokens = cleaned_prompt.split()
        
        # Determine context words (need 2)
        if len(tokens) >= 2:
            current_w1 = tokens[-2]
            current_w2 = tokens[-1]
        elif len(tokens) == 1:
            current_w1 = "<s>"
            current_w2 = tokens[-1]
        else:
            current_w1 = "<s>"
            current_w2 = "<s>"
            
        # Handle OOV - simplistic approach, similar to bigram fallbacks could be added, 
        # but here we rely on smoothing or break if empty.
        
        generated_tokens = []
        for _ in range(max_length):
            if current_w2 == "</s>":
                break
            
            possible_next = self.trigram_counts[(current_w1, current_w2)]
            
            if not possible_next:
                # If we dead end, we could maybe try fallback to bigram?
                # But for strict trigram implementation request:
                break
                
            candidates = list(possible_next.keys())
            counts = list(possible_next.values())
            
            next_word = random.choices(candidates, weights=counts, k=1)[0]
            
            generated_tokens.append(next_word)
            current_w1 = current_w2
            current_w2 = next_word
            
        return prompt + " " + " ".join(generated_tokens)


# --- Example Usage with IMDB Data ---

def main():
    # 1. Load Data
    try:
        df = pd.read_csv('IMDB Dataset.csv')
        print("Dataset loaded successfully.")
        reviews = df['review'].tolist() 
    except FileNotFoundError:
        print("IMDB Dataset.csv not found. Using dummy data or trying small.")
        try:
             df = pd.read_csv('IMDB Dataset_small.csv')
             print("IMDB Dataset_small.csv loaded.")
             reviews = df['review'].tolist()
        except FileNotFoundError:
             reviews = ["The movie was terrible.", "I loved the movie."]

    # 2. Apply Preprocessing
    preprocessor = TextPreprocessor(
        remove_html=True,
        lowercase=True,
        remove_punctuation=False,
        remove_stopwords=False,
        lemmatize=False,
        expand_contractions=True)
        
    tokenized_corpus = []
    print("Preprocessing texts...")
    for r in reviews:
        cleaned_text = preprocessor.process_text(r)
        tokens = cleaned_text.split()
        # Trigram needs two start tokens to have context for the first real word
        tokens = ['<s>', '<s>'] + tokens + ['</s>']
        tokenized_corpus.append(tokens)

    # 3. Split Train/Test
    split_idx = int(len(tokenized_corpus) * 0.8)
    train_data = tokenized_corpus[:split_idx]
    test_data = tokenized_corpus[split_idx:]

    # 4. Initialize and Train
    model = TrigramLanguageModel(alpha=0.01)
    
    start_time = time.time()
    model.train(train_data)
    end_time = time.time()
    print(f"Time to build model: {end_time - start_time:.4f} seconds")

    # 5. Generate Text
    print("\n--- Generated Reviews ---")
    for _ in range(3):
        # We might generate '</s>' at the end, which generate_sentence returns.
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
