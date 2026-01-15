import json
import re

def merge_notebooks():
    # 1. Load the target notebook
    with open('NLP_Project_Notebook_Optimized.ipynb', 'r', encoding='utf-8') as f:
        target_nb = json.load(f)

    # 2. Load the source notebook (preprocessing.ipynb)
    with open('preprocessing.ipynb', 'r', encoding='utf-8') as f:
        source_nb = json.load(f)

    # 3. New Preprocessor Code (matching updated Preprocessor.py)
    new_preprocessor_code = [
        "class TextPreprocessor:\n",
        "    \"\"\"\n",
        "    A robust and professional preprocessing pipeline for NLP tasks.\n",
        "    Designed to handle IMDB movie reviews for Classical, Neural, and Transformer models.\n",
        "    \"\"\"\n",
        "\n",
        "    def __init__(self, \n",
        "                 remove_html: bool = True,\n",
        "                 lowercase: bool = True,\n",
        "                 remove_punctuation: bool = False,\n",
        "                 remove_stopwords: bool = False,\n",
        "                 lemmatize: bool = False,\n",
        "                 expand_contractions: bool = True):\n",
        "        \"\"\"\n",
        "        Initialize the pipeline with specific configuration flags.\n",
        "        \"\"\"\n",
        "        self.remove_html = remove_html\n",
        "        self.lowercase = lowercase\n",
        "        self.remove_punctuation = remove_punctuation\n",
        "        self.remove_stopwords = remove_stopwords\n",
        "        self.lemmatize = lemmatize\n",
        "        self.expand_contractions = expand_contractions\n",
        "\n",
        "        # Pre-load resources to optimize runtime\n",
        "        self.stop_words = set(stopwords.words('english'))\n",
        "        self.stop_words.remove(\"not\")\n",
        "        self.lemmatizer = WordNetLemmatizer()\n",
        "        \n",
        "        # Simple contraction map for expansion\n",
        "        self.contractions_dict = {\n",
        "            \"isn't\": \"is not\", \"aren't\": \"are not\", \"wasn't\": \"was not\", \"weren't\": \"were not\",\n",
        "            \"haven't\": \"have not\", \"hasn't\": \"has not\", \"hadn't\": \"had not\", \"won't\": \"will not\",\n",
        "            \"wouldn't\": \"would not\", \"don't\": \"do not\", \"doesn't\": \"does not\", \"didn't\": \"did not\",\n",
        "            \"can't\": \"cannot\", \"couldn't\": \"could not\", \"shouldn't\": \"should not\", \"mightn't\": \"might not\",\n",
        "            \"mustn't\": \"must not\", \"i'm\": \"i am\", \"you're\": \"you are\", \"he's\": \"he is\", \"she's\": \"she is\",\n",
        "            \"it's\": \"it is\", \"we're\": \"we are\", \"they're\": \"they are\", \"i've\": \"i have\", \"you've\": \"you have\",\n",
        "            \"we've\": \"we have\", \"they've\": \"they have\", \"i'll\": \"i will\", \"you'll\": \"you will\",\n",
        "            \"he'll\": \"he will\", \"she'll\": \"she will\", \"we'll\": \"we will\", \"they'll\": \"they will\"\n",
        "        }\n",
        "        self.contractions_re = re.compile('(%s)' % '|'.join(self.contractions_dict.keys()))\n",
        "        # Normalization Regexes\n",
        "        self.whitespace_re = re.compile(r'\\s+')\n",
        "        self.apostrophe_re = re.compile(r\"[’`´]\")\n",
        "\n",
        "    def _clean_html(self, text: str) -> str:\n",
        "        \"\"\"Removes HTML tags and unescapes HTML entities.\"\"\"\n",
        "        text = html.unescape(text)\n",
        "        clean = re.compile('<.*?>')\n",
        "        return re.sub(clean, ' ', text)\n",
        "\n",
        "    def _expand_contractions(self, text: str) -> str:\n",
        "        \"\"\"Expands common English contractions.\"\"\"\n",
        "        def replace(match):\n",
        "            return self.contractions_dict[match.group(0)]\n",
        "        return self.contractions_re.sub(replace, text)\n",
        "\n",
        "    def _remove_punct(self, text: str) -> str:\n",
        "        \"\"\"Removes punctuation by replacing it with spaces.\"\"\"\n",
        "        return re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)\n",
        "\n",
        "    def _normalize_whitespace(self, text: str) -> str:\n",
        "        \"\"\"Collapse whitespace and strip edges.\"\"\"\n",
        "        return self.whitespace_re.sub(\" \", text).strip()\n",
        "\n",
        "    def _normalize_apostrophes(self, text: str) -> str:\n",
        "        \"\"\"Normalize fancy apostrophes to standard one.\"\"\"\n",
        "        return self.apostrophe_re.sub(\"'\", text)\n",
        "\n",
        "    def process_text(self, text: str) -> Union[str, List[str]]:\n",
        "        \"\"\"\n",
        "        Main execution method.\n",
        "        \"\"\"\n",
        "        if not isinstance(text, str) or not text:\n",
        "            return \"\"\n",
        "\n",
        "        # 1. Cleaning\n",
        "        if self.remove_html:\n",
        "            text = self._clean_html(text)\n",
        "        \n",
        "        # 1.5 Normalization\n",
        "        text = self._normalize_apostrophes(text)\n",
        "        text = self._normalize_whitespace(text)\n",
        "\n",
        "        # 2. Lowercasing\n",
        "        if self.lowercase:\n",
        "            text = text.lower()\n",
        "            \n",
        "        # 3. Expansion\n",
        "        if self.expand_contractions:\n",
        "            text = self._expand_contractions(text)\n",
        "\n",
        "        # 4. Punctuation Removal\n",
        "        if self.remove_punctuation:\n",
        "            text = self._remove_punct(text)\n",
        "\n",
        "        # 5. Tokenization\n",
        "        tokens = word_tokenize(text)\n",
        "\n",
        "        # 6. Stopword Removal\n",
        "        if self.remove_stopwords:\n",
        "            tokens = [w for w in tokens if w not in self.stop_words]\n",
        "\n",
        "        # 7. Lemmatization\n",
        "        if self.lemmatize:\n",
        "            tokens = [self.lemmatizer.lemmatize(w) for w in tokens]\n",
        "\n",
        "        return \" \".join(tokens)\n"
    ]

    # 4. Find and Replace Class Definition
    for cell in target_nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if "class TextPreprocessor" in source_str:
                cell['source'] = new_preprocessor_code
                print("Updated TextPreprocessor class.")
                break

    # 5. Update Data Loading Cell
    data_loading_addon = [
        "\n",
        "# 3. Pipeline for Transformer (Sentiment Classification)\n",
        "print(\"Preprocessing for Transformer models...\")\n",
        "transformer_prep = TextPreprocessor(\n",
        "    remove_html=True, \n",
        "    lowercase=True, \n",
        "    remove_punctuation=False, # BERT-like models benefit from punctuation\n",
        "    remove_stopwords=False, \n",
        "    lemmatize=False,\n",
        "    expand_contractions=True\n",
        ")\n",
        "\n",
        "transformer_reviews = []\n",
        "labels = []\n",
        "\n",
        "if 'sentiment' in df.columns:\n",
        "    sentiment_map = {'positive': 1, 'negative': 0}\n",
        "    labels = df['sentiment'].map(sentiment_map).tolist()[:SAMPLE_SIZE]\n",
        "else:\n",
        "    print(\"Warning: 'sentiment' column not found. Creating dummy labels.\")\n",
        "    labels = [0] * len(raw_reviews) # Dummy\n",
        "\n",
        "for r in raw_reviews:\n",
        "    # process_text returns a cleaned string, which is what we want for Transformers\n",
        "    clean_text = transformer_prep.process_text(r)\n",
        "    transformer_reviews.append(clean_text)\n",
        "\n",
        "train_transformer = transformer_reviews[:split_idx]\n",
        "test_transformer  = transformer_reviews[split_idx:]\n",
        "train_labels = labels[:split_idx]\n",
        "test_labels = labels[split_idx:]\n"
    ]

    for cell in target_nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if "ngram_prep = TextPreprocessor" in source_str and "neural_prep = TextPreprocessor" in source_str:
                # remove any previous transformer block if it exists (simple check)
                if "# 3. Pipeline for Transformer" not in source_str:
                    cell['source'].extend(data_loading_addon)
                    print("Updated Data Loading cell.")
                break

    # 6. Extract Task 4 Cells
    task4_start_index = -1
    for i, cell in enumerate(source_nb['cells']):
        if cell['cell_type'] == 'markdown':
            source_str = "".join(cell['source'])
            if "Task 4: Transformer Fine-tuning" in source_str:
                task4_start_index = i
                break
    
    if task4_start_index != -1:
        task4_cells = source_nb['cells'][task4_start_index:]
        print(f"Found Task 4 starting at cell {task4_start_index}. Extracting {len(task4_cells)} cells.")
        
        # 7. Modify Task 4 Cells to use our new variable names
        # In preprocessing.ipynb Task 4 uses:
        # df["review_clean_transformer"], df["sentiment"]/label
        # We need to map these to: train_transformer, test_transformer, train_labels, test_labels
        
        # We need to rewrite the dataset creation part (Cell 25 in original)
        # Original:
        # df_hf = df.copy()
        # df_hf["label"] = ...
        # hf_dataset = DatasetDict(...)
        
        # New replacement code for the Dataset Creation cell:
        new_dataset_code = [
            "# Create HuggingFace Dataset from our preprocessed lists\n",
            "from datasets import Dataset, DatasetDict\n",
            "\n",
            "train_dataset = Dataset.from_dict({'text': train_transformer, 'label': train_labels})\n",
            "test_dataset = Dataset.from_dict({'text': test_transformer, 'label': test_labels})\n",
            "\n",
            "# Further split train into train/validation (e.g. 90/10 of training set)\n",
            "train_val_split = train_dataset.train_test_split(test_size=0.1, seed=42)\n",
            "\n",
            "hf_dataset = DatasetDict({\n",
            "    'train': train_val_split['train'],\n",
            "    'validation': train_val_split['test'],\n",
            "    'test': test_dataset\n",
            "})\n",
            "\n",
            "print(hf_dataset)\n"
        ]

        # Scan task4_cells to find the dataset creation cell and replace it
        for cell in task4_cells:
            if cell['cell_type'] == 'code':
                source_str = "".join(cell['source'])
                if "hf_dataset = DatasetDict" in source_str:
                    cell['source'] = new_dataset_code
                    print("Replaced DatasetDict creation code.")
        
        # Also need to update tokenizer call (Cell 27 in original)
        # Original uses batch["review_clean_transformer"]
        # New dataset has 'text' column
        for cell in task4_cells:
            if cell['cell_type'] == 'code':
                source_str = "".join(cell['source'])
                if "def tokenize_minilm(batch):" in source_str:
                    cell['source'] = [s.replace('batch["review_clean_transformer"]', 'batch["text"]') for s in cell['source']]
                    print("Updated tokenizer to use 'text' column.")
                if "def tokenize_bpe(batch):" in source_str: # If BPE cell exists
                    cell['source'] = [s.replace('batch["review_clean_transformer"]', 'batch["text"]') for s in cell['source']]

        # Removing "remove_columns" calls that might reference old names if present
        for cell in task4_cells:
            if cell['cell_type'] == 'code':
                cell['source'] = [s.replace('remove_columns(["review_clean_transformer"])', 'remove_columns(["text"])') for s in cell['source']]

        # Append
        target_nb['cells'].extend(task4_cells)
        print("Appended Task 4 cells.")

    else:
        print("Error: Could not find Task 4 in source notebook.")

    # 8. Save
    with open('NLP_Project_Merged.ipynb', 'w', encoding='utf-8') as f:
        json.dump(target_nb, f, indent=1)
    print("Saved merged notebook to NLP_Project_Merged.ipynb")

if __name__ == "__main__":
    merge_notebooks()
