import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import os
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer

# Ensure results directory exists
OUTPUT_DIR = "results/task1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download NLTK data if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger') # For POS tagging
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng') # Newer versions might need this
except LookupError:
    # Just in case the above name is different in some versions or it falls back
    try: 
        nltk.download('averaged_perceptron_tagger_eng') 
    except:
        pass

# Load Data
print("Loading data...")
try:
    df = pd.read_csv("IMDB Dataset.csv")
    print(f"Data Loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'IMDB Dataset.csv' not found.")
    exit(1)

# 1. Class Distribution
print("\n--- Class Distribution ---")
class_counts = df['sentiment'].value_counts()
print(class_counts)

plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df)
plt.title('Class Distribution')
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png")
plt.close()

# 2. Text Length Analysis
print("\n--- Text Length Analysis ---")
df['char_length'] = df['review'].apply(len)
df['word_count'] = df['review'].apply(lambda x: len(x.split()))

print("Average Word Count per Class:")
print(df.groupby('sentiment')['word_count'].mean())

print("Median Word Count per Class:")
print(df.groupby('sentiment')['word_count'].median())

# Plot Length Distribution
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='word_count', hue='sentiment', kde=True, bins=50)
plt.title('Distribution of Review Lengths (Word Count)')
plt.savefig(f"{OUTPUT_DIR}/length_distribution.png")
plt.close()

# 3. Word Analysis
print("\n--- Word Frequency Analysis ---")
stop_words = set(stopwords.words('english'))

def clean_and_tokenize(text):
    # Remove HTML tags (basic)
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase and split
    words = text.lower().split()
    # Remove stopwords
    return [w for w in words if w not in stop_words]

# Process a subset to save time if dataset is huge, but expected size is manageable
# For word clouds/freq analysis, let's sample or just process all if fast enough. 
# 50k reviews is fine.
print("Tokenizing... (this might take a moment)")
df['tokens'] = df['review'].apply(clean_and_tokenize)

def get_most_common_words(tokens_series, n=20):
    all_words = [word for tokens in tokens_series for word in tokens]
    return Counter(all_words).most_common(n)

# Positive
pos_reviews = df[df['sentiment'] == 'positive']['tokens']
pos_common = get_most_common_words(pos_reviews)
print("\nMost Frequent Words (Positive):")
print(pos_common)

# Negative
neg_reviews = df[df['sentiment'] == 'negative']['tokens']
neg_common = get_most_common_words(neg_reviews)
print("\nMost Frequent Words (Negative):")
print(neg_common)

# Word Clouds
print("\nGenerating Word Clouds...")
wc = WordCloud(width=800, height=400, background_color='white')

# Positive WordCloud
pos_text = ' '.join([' '.join(tokens) for tokens in pos_reviews])
wc.generate(pos_text)
wc.to_file(f"{OUTPUT_DIR}/wordcloud_positive.png")

# Negative WordCloud
neg_text = ' '.join([' '.join(tokens) for tokens in neg_reviews])
wc.generate(neg_text)
wc.to_file(f"{OUTPUT_DIR}/wordcloud_negative.png")

# 4. N-gram Analysis (Bi-grams)
print("\n--- N-gram Analysis (Bigrams) ---")
def get_top_grams(corpus, n=None, ngram_range=(2,2)):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# Use a sample for N-grams to avoid memory issues if running on local machine
sample_n = 5000 
subset_pos = df[df['sentiment'] == 'positive']['review'].iloc[:sample_n]
subset_neg = df[df['sentiment'] == 'negative']['review'].iloc[:sample_n]

top_bi_pos = get_top_grams(subset_pos, n=10, ngram_range=(2,2))
print("\nTop Bigrams (Positive - Sample):")
print(top_bi_pos)

top_bi_neg = get_top_grams(subset_neg, n=10, ngram_range=(2,2))
print("\nTop Bigrams (Negative - Sample):")
print(top_bi_neg)

# 5. POS Tagging Analysis
print("\n--- POS Tagging Analysis (Using Sample) ---")
# Use the same subset as N-grams for speed (5000 samples)
# We need the original text or tokens. We have 'tokens' column but we need to tag them.
# Note: nltk.pos_tag expects a list of words.

def get_pos_tags(text):
    # We use the raw text split to keep some context or use our cleaned tokens?
    # Better to use cleaned tokens for consistency, but POS taggers like sentence structure.
    # However, our clean_and_tokenize removed punctuation/stops, which might affect tagger accuracy slightly
    # but is standard for bag-of-words style analysis. Let's use our tokens.
    return nltk.pos_tag(text)

# Flatten the list of lists of tokens for the subset
print("Tagging POS... (this allows us to filter for Adjectives)")

pos_subset_tokens = df[df['sentiment'] == 'positive']['tokens'].iloc[:sample_n]
neg_subset_tokens = df[df['sentiment'] == 'negative']['tokens'].iloc[:sample_n]

def analyze_pos(token_series, label):
    all_tags = []
    adjectives = []
    
    for tokens in token_series:
        tags = nltk.pos_tag(tokens)
        all_tags.extend([tag for word, tag in tags])
        # JJ: Adjective, JJR: Adj comparative, JJS: Adj superlative
        adjectives.extend([word for word, tag in tags if tag.startswith('JJ')])
    
    # POS Counts
    tag_counts = Counter(all_tags)
    
    # Adjective Counts
    adj_counts = Counter(adjectives)
    
    return tag_counts, adj_counts

pos_tags_counts, pos_adj_counts = analyze_pos(pos_subset_tokens, "Positive")
neg_tags_counts, neg_adj_counts = analyze_pos(neg_subset_tokens, "Negative")

# Plot POS Distribution (Top 10 tags)
# Normalize to simple generic tags for display if desired, or keep specific specific tags (NN, VB, JJ, etc)
# Let's plot side by side
pos_common_tags = dict(pos_tags_counts.most_common(10))
neg_common_tags = dict(neg_tags_counts.most_common(10))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(x=list(pos_common_tags.keys()), y=list(pos_common_tags.values()), ax=axes[0], color='blue')
axes[0].set_title('Top 10 POS Tags (Positive)')
sns.barplot(x=list(neg_common_tags.keys()), y=list(neg_common_tags.values()), ax=axes[1], color='red')
axes[1].set_title('Top 10 POS Tags (Negative)')
plt.savefig(f"{OUTPUT_DIR}/pos_distribution.png")
plt.close()

# 6. Adjective Analysis
print("\n--- Adjective Frequency Analysis ---")
print("Most Frequent Adjectives (Positive):")
print(pos_adj_counts.most_common(20))

print("\nMost Frequent Adjectives (Negative):")
print(neg_adj_counts.most_common(20))

# Word Cloud for Adjectives only
wc_adj = WordCloud(width=800, height=400, background_color='white')

# Positive Adjectives
wc_adj.generate_from_frequencies(pos_adj_counts)
wc_adj.to_file(f"{OUTPUT_DIR}/wordcloud_adjectives_positive.png")

# Negative Adjectives
wc_adj.generate_from_frequencies(neg_adj_counts)
wc_adj.to_file(f"{OUTPUT_DIR}/wordcloud_adjectives_negative.png")


print(f"\nAnalysis Complete. Plots saved to '{OUTPUT_DIR}'.")
