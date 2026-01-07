import re
import html
import string
from typing import List, Optional, Union
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading necessary NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')

class TextPreprocessor:
    """
    A robust and professional preprocessing pipeline for NLP tasks.
    Designed to handle IMDB movie reviews for Classical, Neural, and Transformer models.
    """

    def __init__(self, 
                 remove_html: bool = True,
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_stopwords: bool = False,
                 lemmatize: bool = False,
                 expand_contractions: bool = True):
        """
        Initialize the pipeline with specific configuration flags.
        
        Args:
            remove_html (bool): Strip HTML tags (e.g., <br />). Default True.
            lowercase (bool): Convert text to lowercase. Default True.
            remove_punctuation (bool): Remove punctuation characters.
            remove_stopwords (bool): Remove standard English stopwords.
            lemmatize (bool): Apply WordNet lemmatization.
            expand_contractions (bool): Expand "isn't" to "is not".
        """
        self.remove_html = remove_html
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.expand_contractions = expand_contractions

        # Pre-load resources to optimize runtime
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.remove("not")
        self.lemmatizer = WordNetLemmatizer()
        
        # Simple contraction map for expansion
        self.contractions_dict = {
            "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
            "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "can't": "cannot", "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
            "mustn't": "must not", "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
            "it's": "it is", "we're": "we are", "they're": "they are", "i've": "i have", "you've": "you have",
            "we've": "we have", "they've": "they have", "i'll": "i will", "you'll": "you will",
            "he'll": "he will", "she'll": "she will", "we'll": "we will", "they'll": "they will"
        }
        self.contractions_re = re.compile('(%s)' % '|'.join(self.contractions_dict.keys()))

    def _clean_html(self, text: str) -> str:
        """Removes HTML tags and unescapes HTML entities."""
        text = html.unescape(text)
        # Regex for HTML tags
        clean = re.compile('<.*?>')
        return re.sub(clean, ' ', text)

    def _expand_contractions(self, text: str) -> str:
        """Expands common English contractions."""
        def replace(match):
            return self.contractions_dict[match.group(0)]
        return self.contractions_re.sub(replace, text)

    def _remove_punct(self, text: str) -> str:
        """
        Removes punctuation by replacing it with spaces.
        This prevents 'word,word' from becoming 'wordword'.
        """
        # Replace punctuation with a space
        return re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)

    def process_text(self, text: str) -> Union[str, List[str]]:
        """
        Main execution method. Applies enabled steps in the logical order.
        
        Returns:
            str: If the final output is a joined string.
            List[str]: If the processing flow ends in tokenization without re-joining.
        """
        if not isinstance(text, str) or not text:
            return ""

        # 1. Cleaning
        if self.remove_html:
            text = self._clean_html(text)
        
        # 2. Lowercasing
        if self.lowercase:
            text = text.lower()
            
        # 3. Expansion (must be after lowercasing for simple dict matching)
        if self.expand_contractions:
            text = self._expand_contractions(text)

        # 4. Punctuation Removal
        if self.remove_punctuation:
            text = self._remove_punct(text)

        # 5. Tokenization
        # We always tokenize to perform word-level operations (stopword/lemma)
        tokens = word_tokenize(text)

        # 6. Stopword Removal
        if self.remove_stopwords:
            tokens = [w for w in tokens if w not in self.stop_words]

        # 7. Lemmatization
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(w) for w in tokens]

        # Return list of tokens or join back to string depending on downstream need.
        # For this pipeline, we generally return the list of tokens for Classical models,
        # but for compatibility, we will join them back into a clean string 
        # because Tokenizers for Transformers/LSTMs often expect string input 
        # and do their own internal splitting.
        
        return " ".join(tokens)

# --- Usage Example / Demonstration ---

if __name__ == "__main__":
    raw_review = """<br /><br />The movie wasn't good. It was <b>terrible</b>! 
    I can't believe I paid $10 for this garbage..."""

    print("--- RAW TEXT ---")
    print(raw_review)
    print("\n")

    # Configuration 1: For Classical N-gram Models (Aggressive cleaning)
    print("--- CLASSICAL PIPELINE (N-grams) ---")
    classic_pipe = TextPreprocessor(
        remove_html=True,
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        lemmatize=True,
        expand_contractions=True
    )
    print(classic_pipe.process_text(raw_review))

    # Configuration 2: For Deep Learning / Contextual Models (Preserve structure)
    print("\n--- NEURAL PIPELINE (LSTM/Transformer) ---")
    neural_pipe = TextPreprocessor(
        remove_html=True,
        lowercase=True, # Often True for LSTM, Optional for BERT
        remove_punctuation=False, # Punctuation carries meaning
        remove_stopwords=False, # Stopwords carry structure
        lemmatize=False,
        expand_contractions=True
    )
    print(neural_pipe.process_text(raw_review))