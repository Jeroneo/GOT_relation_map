"""
STEP 2 — Cluster raw person names into canonical characters using embeddings.

Strategy: sentence-transformers embeddings + Agglomerative Clustering.

Input:  data/raw_persons.json        (from step 1)
Output: data/canonical_characters.json  — list of character dicts
        data/alias_map.json             — { alias: canonical_name }
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE       = Path("data/raw_persons.json")
OUTPUT_CHARS     = Path("data/canonical_characters.json")
OUTPUT_ALIAS_MAP = Path("data/alias_map.json")

# Embedding model — small but good for names
EMBED_MODEL      = "all-MiniLM-L6-v2"

# Agglomerative clustering threshold (cosine distance 0–2).
# Lower = stricter (fewer merges). Tune this if clusters are too broad/narrow.
DISTANCE_THRESHOLD = 0.35

# After clustering, drop characters mentioned fewer than N times total
MIN_TOTAL_MENTIONS = 10

# Noise prefixes — names starting with these are likely titles, not characters
NOISE_PREFIXES = {
    "the", "my", "your", "his", "her", "our", "their",
    "a", "an", "this", "that", "old", "young", "poor",
    "good", "great", "ser",   # 'ser' alone without a surname is kept later
}

# Noise single tokens (case-insensitive)
NOISE_TOKENS = {
    "lord", "lady", "king", "queen", "prince", "princess",
    "maester", "septon", "septa", "knight", "guard",
    "god", "gods", "stranger", "smith", "maiden", "mother", "father",
    "brother", "sister", "son", "daughter",
}
# ─────────────────────────────────────────────────────────────────────────────


def load_raw_persons(path: Path) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_noise(name: str) -> bool:
    """Return True if the name looks like a title or generic phrase."""
    tokens = name.lower().split()
    if not tokens:
        return True
    if tokens[0] in NOISE_PREFIXES:
        return True
    # Single generic word
    if len(tokens) == 1 and tokens[0] in NOISE_TOKENS:
        return True
    # Contains digits
    if any(c.isdigit() for c in name):
        return True
    # Very short single-token non-capitalised
    if len(tokens) == 1 and len(name) <= 2:
        return True
    return False


def clean_name(name: str) -> str:
    """Strip extra whitespace and quotation marks."""
    name = re.sub(r"[\"''\u2018\u2019\u201c\u201d]", "", name)
    return " ".join(name.split())


def embed_names(names: list[str], model_name: str) -> np.ndarray:
    """Embed a list of names using sentence-transformers."""
    print(f"  Loading embedding model '{model_name}' …")
    model = SentenceTransformer(model_name)
    print(f"  Embedding {len(names)} names …")
    embeddings = model.encode(names, show_progress_bar=True, batch_size=128)
    return embeddings


def cluster_embeddings(
    embeddings: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Agglomerative clustering with cosine distance.
    Returns an array of cluster labels (one per name).
    """
    print(f"  Clustering (threshold={threshold}) …")
    dist_matrix = cosine_distances(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist_matrix)
    n_clusters = labels.max() + 1
    print(f"  → {n_clusters} clusters found")
    return labels


def build_canonical_map(
    names: list[str],
    labels: np.ndarray,
    counts: dict[str, int],
) -> tuple[dict[str, str], list[dict]]:
    """
    For each cluster, pick the canonical name = alias with most mentions.
    Returns:
      alias_map   : { any_alias → canonical_name }
      characters  : list of { id, name, aliases, mentions }
    """
    clusters: dict[int, list[str]] = defaultdict(list)
    for name, label in zip(names, labels):
        clusters[int(label)].append(name)

    alias_map: dict[str, str] = {}
    characters: list[dict] = []

    for label, aliases in clusters.items():
        # Canonical = highest mention count among aliases
        canonical = max(aliases, key=lambda n: counts.get(n, 0))
        total_mentions = sum(counts.get(a, 0) for a in aliases)

        for alias in aliases:
            alias_map[alias] = canonical

        characters.append({
            "id": canonical.lower().replace(" ", "_"),
            "name": canonical,
            "aliases": sorted(set(aliases) - {canonical}),
            "mentions": total_mentions,
        })

    # Sort by mention count descending
    characters.sort(key=lambda c: c["mentions"], reverse=True)
    return alias_map, characters


def main():
    print("── Step 2: Alias Clustering via Embeddings ──\n")

    raw = load_raw_persons(INPUT_FILE)
    print(f"Loaded {len(raw)} raw names from {INPUT_FILE}")

    # ── 1. Clean & filter noise ───────────────────────────────────────────────
    cleaned: dict[str, int] = {}
    for name, count in raw.items():
        name = clean_name(name)
        if not is_noise(name) and name:
            # Accumulate counts (cleaning may merge duplicates)
            cleaned[name] = cleaned.get(name, 0) + count

    print(f"After noise filtering: {len(cleaned)} names")

    # ── 2. Embed ──────────────────────────────────────────────────────────────
    names = list(cleaned.keys())
    embeddings = embed_names(names, EMBED_MODEL)

    # ── 3. Cluster ────────────────────────────────────────────────────────────
    labels = cluster_embeddings(embeddings, DISTANCE_THRESHOLD)

    # ── 4. Build canonical map ────────────────────────────────────────────────
    alias_map, characters = build_canonical_map(names, labels, cleaned)

    # ── 5. Filter by total mentions ───────────────────────────────────────────
    characters = [c for c in characters if c["mentions"] >= MIN_TOTAL_MENTIONS]
    # Rebuild alias_map to only include characters that passed the filter
    surviving = {c["name"] for c in characters}
    alias_map = {a: c for a, c in alias_map.items() if c in surviving}

    # ── 6. Report ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"Characters after filtering (>= {MIN_TOTAL_MENTIONS} mentions): {len(characters)}")
    print(f"\nTop 30 characters:")
    for char in characters[:30]:
        aliases_str = ", ".join(char["aliases"][:4])
        if len(char["aliases"]) > 4:
            aliases_str += f" (+{len(char['aliases'])-4} more)"
        print(f"  {char['mentions']:5d}  {char['name']:<28}  [{aliases_str}]")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    Path("data").mkdir(exist_ok=True)
    with open(OUTPUT_CHARS, "w", encoding="utf-8") as f:
        json.dump(characters, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_ALIAS_MAP, "w", encoding="utf-8") as f:
        json.dump(alias_map, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Characters saved to {OUTPUT_CHARS}")
    print(f"✓ Alias map saved to  {OUTPUT_ALIAS_MAP}")


if __name__ == "__main__":
    main()
