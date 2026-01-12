# CSV Annotation Helper

A lightweight, interactive terminal tool to annotate sentiment labels and confidence for human annotators. Works with the existing columns in your dataset and adds/updates `human_label_annotator{1|2}` and `human_confidence_annotator{1|2}`.

## Quick Start

1. Activate your Python environment.
2. Run the annotator:

```bash
python annotate_csv.py --input human_eval_100.csv --annotator 1
```

This will create `human_eval_100.annotated.csv` and save after each annotation.

## Options

- `--annotator {1|2}`: choose annotator slot (defaults to 1).
- `--output <path>`: write to a specific file (default: `<input>.annotated.csv`).
- `--inplace`: write back into the input file.
- `--start-index <id>`: start at specific `sample_index` (or row number).
- `--dry-run N`: show the next N samples without annotating.

## Controls

- `p` → positive
- `n` → negative
- `s` → skip
- `b` → go back to previous sample
- `q` → quit and save

Confidence can be entered as `0.0–1.0` or `1–5` (mapped to 0–1).

## Tips

- Progress is saved automatically after each sample.
- Use `--start-index` to resume at a specific `sample_index`.
- Use `--annotator 2` to fill the second annotator columns.
