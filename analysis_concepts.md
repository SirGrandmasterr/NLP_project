# Analysis Concepts and Future Directions

This document outlines the theoretical concepts behind the data exploration performed in Task 1 and suggests further analyses to deepen understanding of the IMDB dataset.

## 1. Concepts Behind Implemented Analyses

### Class Distribution
*   **Concept**: Checking for **Class Imbalance**.
*   **Why**: Machine learning models tend to be biased towards the majority class. If 90% of reviews were positive, a model could achieve 90% accuracy by simply predicting "positive" for everything, without learning anything useful.
*   **Finding**: The 50/50 split means we can use standard metrics (Accuracy, F1-score) without needing resampling techniques (SMOTE, undersampling).

### Text Length Analysis
*   **Concept**: **Feature Engineering** and **Proxy for Complexity**.
*   **Why**: Sometimes, one class is significantly longer (e.g., angry rants might be longer than happy compliments, or vice versa). Length can be a simple but powerful feature. Extreme outliers (very short or very long texts) might also need to be pruned to prevent memory issues in neural networks.
*   **Finding**: Lengths are nearly identical, so length is not a predictive feature here.

### Word Frequency (Unigrams) & Word Clouds
*   **Concept**: **Lexical Profiling**.
*   **Why**: Identifies the specific vocabulary associated with each class. It also reveals "Stopwords" (common words like "the", "and") that carry little meaning and domain-specific stopwords (like "movie" or "film" in this dataset) that appear everywhere and might be noise.
*   **Finding**: High overlap in vocabulary ("good" appears often in negative reviews), indicating the need for context-aware models.

### N-gram Analysis (Bigrams)
*   **Concept**: **Contextual Dependency**.
*   **Why**: Single words (Unigrams) lose context. "Not good" is negative, but "good" is positive. Bigrams (pairs of words) capture short-range context and common phrases ("special effects", "waste [of] time").
*   **Finding**: Revealed data quality issues (`br br`) that unigram analysis missed.

---

## 2. Suggestions for Further Analysis

### A. Sentiment Consistency / Subjectivity Analysis
*   **What**: Use a pre-trained lexicon (like TextBlob or VADER) to score the "subjectivity" and "polarity" of the reviews.
*   **Why**: Check if the labels match the linguistic sentiment. Are there "positive" reviews that use sarcastic or negative language? This helps identify "hard" samples.

### B. Topic Modeling (LDA or BERTopic)
*   **What**: Unsupervised learning to group reviews into abstract "topics" (e.g., "Horror/Gore", "Romantic Comedy", "Acting/Cast", "Plot Holes").
*   **Why**: Does sentiment correlate with genre? Are horror movies reviewed differently than comedies?

### C. Part-of-Speech (POS) Tagging
*   **What**: Count the ratio of Adjectives vs. Nouns vs. Verbs.
*   **Why**: Sentiment is often carried by adjectives ("terrible", "amazing"). A higher density of adjectives might correlate with stronger sentiment intensity.

### D. Named Entity Recognition (NER)
*   **What**: Extract names of people, locations, and organizations.
*   **Why**: Determine if specific actors or directors are strongly correlated with positive or negative sentiment (e.g., "Meryl Streep" vs. "Uwe Boll").

### E. Readability Scores
*   **What**: Calculate Flesch-Kincaid or similar readability scores.
*   **Why**: Are positive reviews written more simply or more complexly than negative ones?

### F. Embedding Visualization (t-SNE / PCA)
*   **What**: Convert a subset of texts into dense vectors (using BERT or TF-IDF) and project them into 2D space.
*   **Why**: Visually check if positive and negative reviews form distinct clusters. If they overlap heavily, linear models might struggle.
