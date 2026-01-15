import json
import re

def refine_notebook(notebook_path='NLP_Project_Merged.ipynb'):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    
    # regex to find separator lines we can split on
    # e.g. # ----------- or # ======= usually denoting sections
    # avoiding splitting inside functions/classes if indented (simple heuristic)
    split_pattern = re.compile(r'^#\s*[-=]{10,}', re.MULTILINE)

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            new_cells.append(cell)
            continue

        source = "".join(cell['source'])
        
        # If cell is short, keep it
        if len(cell['source']) < 20: 
            new_cells.append(cell)
            continue

        # Check for separators
        # We find their indices basically
        # The logic: split string by the separators, but keep separators attached to the *following* block usually?
        # Or just split by headers.
        
        # Helper to check if a line is a separator
        lines = cell['source']
        chunks = []
        current_chunk = []
        
        for line in lines:
            # Check if this line is a major separator (start of line)
            # and we have amassed some content already
            if (line.startswith("# ----") or line.startswith("# ====")) and len(current_chunk) > 5:
                # Flush current chunk
                chunks.append(current_chunk)
                current_chunk = []
            
            current_chunk.append(line)
        
        if current_chunk:
            chunks.append(current_chunk)

        # Allow splitting if we found chunks
        if len(chunks) > 1:
            print(f"Splitting a cell into {len(chunks)} parts.")
            for chunk in chunks:
                new_cell = {
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': chunk
                }
                new_cells.append(new_cell)
        else:
            new_cells.append(cell)

    nb['cells'] = new_cells

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Refined notebook saved to {notebook_path}")

if __name__ == "__main__":
    refine_notebook()
