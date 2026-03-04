"""
STEP 3 — Co-occurrence extraction + relationship type classification.

Slides a window over each book and counts how often every pair of
canonical characters appears together. Then classifies the dominant
relationship type (family / ally / enemy / romance / sworn / other)
from the surrounding text.

Input:  books/*.txt
        data/alias_map.json
        data/canonical_characters.json
Output: data/edges.json   — list of edge dicts
"""

import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BOOKS_DIR      = Path("books")
ALIAS_MAP_FILE = Path("data/alias_map.json")
CHARS_FILE     = Path("data/canonical_characters.json")
OUTPUT_FILE    = Path("data/edges.json")

# How many words to consider a "scene" between two characters
WINDOW_WORDS   = 100

# Minimum co-occurrence count to keep an edge
MIN_WEIGHT     = 5

# ── Relation keywords (extend freely) ────────────────────────────────────────
RELATION_KEYWORDS: dict[str, list[str]] = {
    "family": [
        "father", "mother", "son", "daughter", "brother", "sister",
        "husband", "wife", "uncle", "aunt", "cousin", "nephew", "niece",
        "wed", "wedding", "married", "born", "birth", "bastard", "heir",
        "grandfather", "grandmother", "kin", "blood", "family", "House",
    ],
    "enemy": [
        "killed", "kill", "murder", "murdered", "enemy", "enemies",
        "betrayed", "betrayal", "traitor", "treason", "hated", "hate",
        "fought", "fight", "slew", "slay", "death", "execute", "hanged",
        "war", "battle", "rival", "revenge", "vengeance", "sword",
        "captured", "prisoner", "hostage", "attacked", "threatened",
    ],
    "ally": [
        "ally", "alliance", "allied", "trusted", "trust", "loyal",
        "loyalty", "served", "serve", "helped", "help", "friend",
        "friendship", "together", "joined", "joined forces", "supported",
        "support", "agreement", "deal", "pact", "banner",
    ],
    "romance": [
        "love", "loved", "lover", "kiss", "kissed", "heart", "beloved",
        "desire", "desired", "passion", "bed", "bedded", "wed", "wedded",
        "betrothed", "betrothal", "affection", "longing", "beauty",
        "beautiful", "handsome",
    ],
    "sworn": [
        "sworn sword", "sworn shield", "kingsguard", "queensguard",
        "vow", "oath", "pledged", "pledge", "knelt", "kneel",
        "master", "protector", "bodyguard", "steward", "squire",
        "served", "in service", "commanded", "orders",
    ],
}
# ─────────────────────────────────────────────────────────────────────────────


def load_alias_map(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_characters(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_regex_map(alias_map: dict[str, str]) -> list[tuple[re.Pattern, str]]:
    """
    Build a list of (compiled regex, canonical_name) pairs.
    Longer aliases first so 'Jon Snow' matches before 'Jon'.
    """
    pairs = sorted(alias_map.items(), key=lambda x: -len(x[0]))
    compiled = []
    for alias, canonical in pairs:
        # Word-boundary match, case-insensitive
        pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
        compiled.append((pattern, canonical))
    return compiled


def find_chars_in_window(window: str, regex_map) -> set[str]:
    """Return the set of canonical character names found in `window`."""
    found = set()
    for pattern, canonical in regex_map:
        if pattern.search(window):
            found.add(canonical)
    return found


def classify_relation(window: str) -> str:
    """Score each relation type by keyword hits; return the winner."""
    w = window.lower()
    scores: Counter = Counter()
    for rel_type, keywords in RELATION_KEYWORDS.items():
        for kw in keywords:
            if kw in w:
                scores[rel_type] += 1
    if not scores:
        return "other"
    return scores.most_common(1)[0][0]


def extract_edges_from_text(
    text: str,
    regex_map,
    window_words: int = WINDOW_WORDS,
) -> tuple[Counter, dict[tuple, Counter]]:
    """
    Slide a half-overlapping word window over `text`.
    Returns:
      pair_counts : Counter  { (charA, charB) → total co-occurrences }
      pair_types  : dict     { (charA, charB) → Counter of relation types }
    """
    words = text.split()
    step  = window_words // 2

    pair_counts: Counter = Counter()
    pair_types: dict[tuple, Counter] = defaultdict(Counter)

    for i in range(0, len(words) - window_words, step):
        window = " ".join(words[i : i + window_words])
        chars  = find_chars_in_window(window, regex_map)

        if len(chars) < 2:
            continue

        rel = classify_relation(window)

        for a, b in combinations(sorted(chars), 2):
            pair = (a, b)
            pair_counts[pair] += 1
            pair_types[pair][rel] += 1

    return pair_counts, pair_types


def main():
    print("── Step 3: Co-occurrence & Relation Extraction ──\n")

    alias_map  = load_alias_map(ALIAS_MAP_FILE)
    characters = load_characters(CHARS_FILE)
    char_names = {c["name"] for c in characters}

    # Keep only aliases that resolve to a surviving character
    alias_map = {a: c for a, c in alias_map.items() if c in char_names}
    regex_map = build_regex_map(alias_map)
    print(f"Tracking {len(char_names)} characters via {len(alias_map)} aliases\n")

    total_counts: Counter = Counter()
    total_types: dict[tuple, Counter] = defaultdict(Counter)

    for book_path in sorted(Path(BOOKS_DIR).glob("*.txt")):
        print(f"Processing {book_path.name} …")
        text = book_path.read_text(encoding="utf-8", errors="replace")
        counts, types = extract_edges_from_text(text, regex_map)

        total_counts.update(counts)
        for pair, type_counter in types.items():
            total_types[pair].update(type_counter)

        print(f"  → {sum(1 for v in counts.values() if v >= MIN_WEIGHT)} "
              f"edges (>= {MIN_WEIGHT} co-occurrences) in this book")

    # ── Build edge list ───────────────────────────────────────────────────────
    edges = []
    for (source, target), weight in total_counts.items():
        if weight < MIN_WEIGHT:
            continue
        dominant_type = total_types[(source, target)].most_common(1)[0][0]
        type_breakdown = dict(total_types[(source, target)])

        edges.append({
            "source":         source,
            "target":         target,
            "weight":         weight,
            "type":           dominant_type,
            "type_breakdown": type_breakdown,
        })

    # Sort by weight
    edges.sort(key=lambda e: e["weight"], reverse=True)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"Total edges (weight >= {MIN_WEIGHT}): {len(edges)}")

    type_counts = Counter(e["type"] for e in edges)
    print("\nEdges by relation type:")
    for rel, cnt in type_counts.most_common():
        print(f"  {rel:<10} {cnt}")

    print("\nTop 20 strongest relationships:")
    for e in edges[:20]:
        print(f"  {e['weight']:5d}  {e['source']:<28} ↔  {e['target']:<28}  [{e['type']}]")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Edges saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
