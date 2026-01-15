import json
import re

def fix_notebook(notebook_path='NLP_Project_Merged.ipynb'):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    count = 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                # Replace ex["review_clean_transformer"] or batch["review_clean_transformer"]
                # with ["text"]
                if '"review_clean_transformer"' in line or "'review_clean_transformer'" in line:
                    line = line.replace('"review_clean_transformer"', '"text"')
                    line = line.replace("'review_clean_transformer'", "'text'")
                    count += 1
                new_source.append(line)
            cell['source'] = new_source

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Fixed {count} occurrences of 'review_clean_transformer' in {notebook_path}")

if __name__ == "__main__":
    fix_notebook()
