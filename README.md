# ASOIAF Character Relationship Map
## Full pipeline using Embedding Similarity (Strategy B)

---

## Folder structure

```
got_pipeline/
├── books/                  ← PUT YOUR .txt BOOK FILES HERE
│   ├── 1_agot.txt
│   ├── 2_acok.txt
│   ├── 3_asos.txt
│   ├── 4_affc.txt
│   └── 5_adwd.txt
│
├── data/                   ← auto-created by the scripts
│   ├── raw_persons.json
│   ├── canonical_characters.json
│   ├── alias_map.json
│   └── graph.json          ← final output fed to the visualiser
│
├── 01_extract_persons.py
├── 02_cluster_aliases.py
├── 03_extract_edges.py
├── 04_build_graph.py
├── 05_visualise.html
└── README.md
```

---

## 1. Install dependencies

```bash
pip install spacy sentence-transformers scikit-learn networkx rapidfuzz
python -m spacy download en_core_web_lg
```

---

## 2. Add your books

Place plain-text `.txt` files of each book in the `books/` folder.
If you have ePubs, convert them with Calibre:
  File → Export → TXT

---

## 3. Run the pipeline

Each script reads from `data/` and writes back to `data/`.

```bash
# Step 1 — NER extraction (~5–20 min depending on book size)
python 01_extract_persons.py

# Step 2 — Embedding clustering (~2–5 min)
python 02_cluster_aliases.py

# Step 3 — Co-occurrence + relation classification (~5–15 min)
python 03_extract_edges.py

# Step 4 — House inference + centrality + export
python 04_build_graph.py
```

---

## 4. Serve & view

The visualiser loads `data/graph.json` via fetch(), so you need a local server:

```bash
# Python (simplest)
python3 -m http.server 8000

# Or Node
npx serve .
```

Then open: http://localhost:8000/05_visualise.html

---

## Tuning parameters

| File | Variable | Effect |
|------|----------|--------|
| `01` | `MIN_RAW_MENTIONS` | Raise to reduce noise names |
| `02` | `DISTANCE_THRESHOLD` | Lower (e.g. 0.25) = stricter clusters; raise (0.45) = more aggressive merging |
| `02` | `MIN_TOTAL_MENTIONS` | Raise to keep only major characters |
| `03` | `WINDOW_WORDS` | Larger = more scene-level co-occurrences |
| `03` | `MIN_WEIGHT` | Raise to show only strong relationships |

---

## How the embedding clustering works

1. Every raw name ("Jon", "Jon Snow", "Lord Snow") is encoded into a
   high-dimensional vector by `all-MiniLM-L6-v2`.
2. Cosine distance is computed between every pair of names.
3. Agglomerative clustering groups names whose vectors are within
   `DISTANCE_THRESHOLD` of each other (average linkage).
4. The most-mentioned alias in each cluster becomes the canonical name.

This works because names that are semantically or phonetically similar
end up close in the embedding space — no hand-crafted rules needed.
