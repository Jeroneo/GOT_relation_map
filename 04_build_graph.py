"""
STEP 4 — Enrich characters (infer house, compute centrality) and export
         a single graph JSON ready for D3.js.

Input:  data/canonical_characters.json
        data/edges.json
        books/*.txt
Output: data/graph.json   — { nodes: [...], links: [...] }
"""

import json
import re
from collections import Counter
from pathlib import Path

import networkx as nx

# ── Config ────────────────────────────────────────────────────────────────────
CHARS_FILE   = Path("data/canonical_characters.json")
EDGES_FILE   = Path("data/edges.json")
BOOKS_DIR    = Path("books")
OUTPUT_FILE  = Path("data/graph.json")

# Known house surnames — used for automatic house inference
HOUSE_NAMES = [
    "Stark", "Lannister", "Targaryen", "Baratheon", "Tyrell",
    "Greyjoy", "Martell", "Tully", "Arryn", "Bolton", "Frey",
    "Mormont", "Karstark", "Umber", "Reed", "Manderly",
    "Clegane", "Seaworth", "Dondarrion", "Florent",
]

HOUSE_COLORS = {
    "stark":      "#7a8fa6",
    "lannister":  "#c9a84c",
    "targaryen":  "#b22222",
    "baratheon":  "#6b8c5a",
    "tyrell":     "#4a7c59",
    "greyjoy":    "#3d5a6b",
    "martell":    "#b8620a",
    "tully":      "#2a5fa8",
    "arryn":      "#3a6b8a",
    "bolton":     "#8b2a2a",
    "frey":       "#8a8a6a",
    "mormont":    "#5a7a6a",
    "clegane":    "#6a4a4a",
    "seaworth":   "#4a6a8a",
    "unknown":    "#666688",
}
# ─────────────────────────────────────────────────────────────────────────────


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def infer_house_from_name(char_name: str) -> str:
    """
    If the character's canonical name contains a known house surname,
    use that directly (e.g. 'Tyrion Lannister' → 'lannister').
    """
    for house in HOUSE_NAMES:
        if house.lower() in char_name.lower():
            return house.lower()
    return None


def infer_house_from_text(
    char_name: str,
    first_token: str,
    all_text: str,
    window: int = 300,
) -> str:
    """
    Search sentences containing the character's first name;
    count which house word appears most often nearby.
    Returns the best-matching house key or 'unknown'.
    """
    house_counts: Counter = Counter()
    pattern = re.compile(r'\b' + re.escape(first_token) + r'\b', re.IGNORECASE)

    for match in pattern.finditer(all_text):
        start = max(0, match.start() - window)
        end   = min(len(all_text), match.end() + window)
        context = all_text[start:end]
        for house in HOUSE_NAMES:
            if re.search(r'\b' + house + r'\b', context):
                house_counts[house.lower()] += 1

    if house_counts:
        return house_counts.most_common(1)[0][0]
    return "unknown"


def compute_centrality(
    characters: list[dict],
    edges: list[dict],
) -> dict[str, dict]:
    """
    Build a NetworkX graph and compute degree + betweenness centrality.
    Returns { canonical_name → { degree, betweenness } }
    """
    G = nx.Graph()
    for c in characters:
        G.add_node(c["name"])
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e["weight"])

    degree      = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

    return {
        name: {
            "degree":      round(degree.get(name, 0), 4),
            "betweenness": round(betweenness.get(name, 0), 4),
        }
        for name in G.nodes
    }


def main():
    print("── Step 4: Enrich & Export Graph JSON ──\n")

    characters = load_json(CHARS_FILE)
    edges      = load_json(EDGES_FILE)

    # ── Load all book text for house inference ────────────────────────────────
    print("Loading books for house inference …")
    all_text = ""
    for path in sorted(Path(BOOKS_DIR).glob("*.txt")):
        all_text += path.read_text(encoding="utf-8", errors="replace") + "\n"
    print(f"  Total text: {len(all_text):,} chars\n")

    # ── Infer house for each character ────────────────────────────────────────
    print("Inferring houses …")
    for char in characters:
        # 1. Try surname match first (fast & reliable)
        house = infer_house_from_name(char["name"])
        if not house:
            # 2. Fallback: scan surrounding text
            first_token = char["name"].split()[0]
            house = infer_house_from_text(char["name"], first_token, all_text)
        char["house"]  = house
        char["color"]  = HOUSE_COLORS.get(house, HOUSE_COLORS["unknown"])

    # ── Compute graph centrality ───────────────────────────────────────────────
    print("Computing centrality metrics …")
    centrality = compute_centrality(characters, edges)
    for char in characters:
        c = centrality.get(char["name"], {})
        char["degree"]      = c.get("degree", 0)
        char["betweenness"] = c.get("betweenness", 0)

    # ── Build D3-ready structure ───────────────────────────────────────────────
    # D3 force graph expects nodes with unique id and links with source/target
    nodes = [
        {
            "id":          char["name"],      # D3 uses this to match links
            "name":        char["name"],
            "house":       char["house"],
            "color":       char["color"],
            "mentions":    char["mentions"],
            "aliases":     char["aliases"],
            "degree":      char["degree"],
            "betweenness": char["betweenness"],
            # Node size hint for the visualiser (log-scaled)
            "size": max(6, min(30, 6 + char["mentions"] ** 0.45)),
        }
        for char in characters
    ]

    links = [
        {
            "source":         e["source"],
            "target":         e["target"],
            "weight":         e["weight"],
            "type":           e["type"],
            "type_breakdown": e["type_breakdown"],
        }
        for e in edges
    ]

    graph = {"nodes": nodes, "links": links}

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"Nodes : {len(nodes)}")
    print(f"Links : {len(links)}")

    house_dist = Counter(n["house"] for n in nodes)
    print("\nCharacters by house:")
    for house, cnt in house_dist.most_common():
        print(f"  {house:<15} {cnt}")

    print("\nTop 10 by betweenness centrality (most 'pivotal'):")
    top_bc = sorted(nodes, key=lambda n: n["betweenness"], reverse=True)[:10]
    for n in top_bc:
        print(f"  {n['betweenness']:.4f}  {n['name']}")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Graph saved to {OUTPUT_FILE}")
    print("   → Feed this file into 05_visualise.html")


if __name__ == "__main__":
    main()
