#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import textwrap
from typing import List, Dict, Optional, Tuple

LABELS = ["positive", "negative"]

WRAP_WIDTH = 100


def read_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, reader.fieldnames or []


def ensure_columns(fieldnames: List[str], annotator: int) -> List[str]:
    label_col = f"human_label_annotator{annotator}"
    conf_col = f"human_confidence_annotator{annotator}"
    updated = list(fieldnames)
    if label_col not in updated:
        updated.append(label_col)
    if conf_col not in updated:
        updated.append(conf_col)
    return updated


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # Ensure all expected keys exist
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)
    os.replace(tmp_path, path)


def format_text(text: str) -> str:
    return "\n".join(textwrap.fill(line, width=WRAP_WIDTH) for line in text.splitlines())


def parse_confidence(inp: str) -> Optional[float]:
    try:
        val = float(inp)
        if 0.0 <= val <= 1.0:
            return val
        # allow 1-5 scale
        if 1 <= val <= 5:
            return (val - 1) / 4.0
    except ValueError:
        return None
    return None


def next_unannotated_index(rows: List[Dict[str, str]], annotator: int, start_from: Optional[int] = None) -> int:
    label_col = f"human_label_annotator{annotator}"
    # Prefer matching by sample_index if provided
    if start_from is not None:
        for i, r in enumerate(rows):
            try:
                if int(r.get("sample_index", -1)) == start_from:
                    return i
            except Exception:
                # fallback to row number
                pass
        # If sample_index not found, treat start_from as row number (0-based)
        if 0 <= start_from < len(rows):
            return start_from
    for i, r in enumerate(rows):
        val = (r.get(label_col, "") or "").strip()
        if not val:
            return i
    return len(rows)


def print_sample(r: Dict[str, str], annotator: int, index: int, total: int) -> None:
    print("\n" + "=" * 80)
    print(f"Sample {index + 1}/{total}")
    sid = r.get("sample_index", "")
    if sid:
        print(f"sample_index: {sid}")
    text = r.get("text", "").strip()
    print("\nText:\n")
    print(format_text(text))
    # Hide gold/model labels during annotation to avoid bias
    label_col = f"human_label_annotator{annotator}"
    conf_col = f"human_confidence_annotator{annotator}"
    if (r.get(label_col, "") or "").strip():
        print(f"\nExisting annotation:")
        print(f"- {label_col}: {r.get(label_col)}")
    if (r.get(conf_col, "") or "").strip():
        print(f"- {conf_col}: {r.get(conf_col)}")


def prompt_label() -> Optional[str]:
    print("\nLabel options: [p]ositive, [n]egative, [s]kip, [q]uit, [b]ack")
    while True:
        inp = input("Enter label (p/n/s/q/b): ").strip().lower()
        if inp in ("p", "positive"):
            return "positive"
        if inp in ("n", "negative"):
            return "negative"
        if inp in ("s", "skip"):
            return None
        if inp in ("q", "quit"):
            return "__quit__"
        if inp in ("b", "back"):
            return "__back__"
        print("Invalid input. Try again: p/n/s/q/b")


def prompt_confidence() -> Optional[float]:
    print("\nConfidence: enter 0-1 (e.g., 0.8) or 1-5 (e.g., 4)")
    while True:
        inp = input("Enter confidence (or press Enter to use 0.5): ").strip()
        if not inp:
            return 0.5
        val = parse_confidence(inp)
        if val is not None:
            return round(val, 3)
        print("Invalid confidence. Use 0-1 or 1-5 scale.")


def annotate_rows(rows: List[Dict[str, str]], fieldnames: List[str], out_path: str, annotator: int, start_from: Optional[int] = None, dry_run: int = 0) -> None:
    label_col = f"human_label_annotator{annotator}"
    conf_col = f"human_confidence_annotator{annotator}"
    total = len(rows)
    idx = next_unannotated_index(rows, annotator, start_from)
    if dry_run > 0:
        end = min(idx + dry_run, total)
        for i in range(idx, end):
            print_sample(rows[i], annotator, i, total)
        return

    history = []  # stack of indices annotated for back navigation

    while idx < total:
        r = rows[idx]
        print_sample(r, annotator, idx, total)
        choice = prompt_label()
        if choice == "__quit__":
            print("\nQuitting. Progress saved.")
            break
        if choice == "__back__":
            if history:
                idx = history.pop()
                print("\nMoved back to previous sample.")
                continue
            else:
                print("\nNo previous sample to go back to.")
                continue
        if choice is None:
            # skip without changing values
            print("Skipped.")
            idx += 1
            continue
        conf = prompt_confidence()
        r[label_col] = choice
        r[conf_col] = str(conf)
        write_csv(out_path, rows, fieldnames)
        print(f"Saved to {out_path}.")
        history.append(idx)
        idx += 1

    # Final save to be safe
    write_csv(out_path, rows, fieldnames)


def default_output(input_path: str, inplace: bool) -> str:
    if inplace:
        return input_path
    base, ext = os.path.splitext(input_path)
    return f"{base}.annotated{ext or '.csv'}"


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CSV annotator for sentiment labels and confidence.")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input CSV (e.g., human_eval_100.csv)")
    parser.add_argument(
        "--output", "-o", help="Path to output CSV. Defaults to <input>.annotated.csv")
    parser.add_argument("--annotator", "-a", type=int, default=1,
                        choices=[1, 2], help="Annotator number: 1 or 2")
    parser.add_argument("--start-index", type=int, default=None,
                        help="Start at specific sample_index or row number")
    parser.add_argument("--inplace", action="store_true",
                        help="Write annotations back to input file")
    parser.add_argument("--dry-run", type=int, default=0,
                        help="Show N upcoming samples without annotating")

    args = parser.parse_args()
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}")
        sys.exit(1)

    rows, fieldnames = read_csv(input_path)
    if not rows:
        print("No rows found in input CSV.")
        sys.exit(1)

    fieldnames = ensure_columns(fieldnames, args.annotator)
    out_path = args.output or default_output(input_path, args.inplace)

    # Auto-resume: if an annotated output already exists (and is different from input),
    # load it to continue from where you left off. Otherwise, seed the output.
    if os.path.exists(out_path) and out_path != input_path:
        try:
            ann_rows, ann_fields = read_csv(out_path)
            # prefer the annotated file's schema and content
            fieldnames = ensure_columns(ann_fields, args.annotator)
            rows = ann_rows
            resume_idx = next_unannotated_index(
                rows, args.annotator, args.start_index)
            # Inform the user about resume position
            sid = None
            if 0 <= resume_idx < len(rows):
                sid = rows[resume_idx].get("sample_index", "")
            print(f"Resuming from existing annotations in {out_path}. Next index: {resume_idx + 1}/{len(rows)}" + (
                f" (sample_index={sid})" if sid else ""))
        except Exception as e:
            print(
                f"Warning: Could not read existing annotated file '{out_path}' ({e}). Seeding fresh output.")
            write_csv(out_path, rows, fieldnames)
    else:
        # If output path differs and doesn't exist yet, or we are writing in-place, seed it
        write_csv(out_path, rows, fieldnames)

    annotate_rows(rows, fieldnames, out_path, args.annotator,
                  args.start_index, args.dry_run)

    print("\nAll done.")


if __name__ == "__main__":
    main()
