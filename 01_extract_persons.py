"""
STEP 1 — Extract all PERSON entities from the books using spaCy NER.

Input:  books/*.txt  (plain text files of each ASOIAF book)
Output: data/raw_persons.json  — { "name": count, ... }
"""

import json
import os
from collections import Counter
from pathlib import Path

import spacy

# ── Config ────────────────────────────────────────────────────────────────────
BOOKS_DIR   = Path("books")          # folder containing .txt files
OUTPUT_DIR  = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "raw_persons.json"
CHUNK_SIZE  = 100_000                # characters per spaCy chunk (memory limit)
MIN_RAW_MENTIONS = 3                 # drop names seen fewer than N times total
# ─────────────────────────────────────────────────────────────────────────────

def load_books(books_dir: Path) -> dict[str, str]:
    """Load every .txt file in the books directory."""
    texts = {}
    for path in sorted(books_dir.glob("*.txt")):
        print(f"  Loading {path.name} …")
        texts[path.stem] = path.read_text(encoding="utf-8", errors="replace")
    return texts


def extract_persons_from_text(text: str, nlp) -> Counter:
    """
    Run spaCy NER over `text` in chunks.
    Returns a Counter of raw entity strings labelled PERSON.
    """
    persons: Counter = Counter()
    total = len(text)

    for start in range(0, total, CHUNK_SIZE):
        chunk = text[start : start + CHUNK_SIZE]
        doc = nlp(chunk)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                # Basic sanity: at least 2 chars, not purely punctuation
                if len(name) >= 2 and any(c.isalpha() for c in name):
                    persons[name] += 1

        pct = min(start + CHUNK_SIZE, total) / total * 100
        print(f"    {pct:5.1f}%", end="\r", flush=True)

    print()
    return persons


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading spaCy model (en_core_web_lg) …")
    nlp = spacy.load("en_core_web_lg")
    # Disable components we don't need for speed
    nlp.select_pipes(enable=["tok2vec", "ner"])

    print(f"\nScanning books in '{BOOKS_DIR}/' …")
    books = load_books(BOOKS_DIR)
    if not books:
        raise FileNotFoundError(
            f"No .txt files found in '{BOOKS_DIR}/'. "
            "Put your book files there and re-run."
        )

    all_persons: Counter = Counter()

    for book_name, text in books.items():
        print(f"\n[{book_name}]  ({len(text):,} chars)")
        persons = extract_persons_from_text(text, nlp)
        print(f"  → {len(persons)} unique raw names, "
              f"{sum(persons.values())} total mentions")
        all_persons.update(persons)

    # Filter by minimum mentions
    filtered = {
        name: count
        for name, count in all_persons.items()
        if count >= MIN_RAW_MENTIONS
    }

    print(f"\n{'─'*50}")
    print(f"Total unique raw names (>= {MIN_RAW_MENTIONS} mentions): {len(filtered)}")
    print(f"\nTop 30 most mentioned:")
    for name, count in Counter(filtered).most_common(30):
        print(f"  {count:5d}  {name}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
