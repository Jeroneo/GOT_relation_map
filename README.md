# GOT Character Feature Extractor, Search Engine & Relationship Network

This project parses HTML pages from A Wiki of Ice and Fire (AWOIAF), extracts rich features for Game of Thrones characters, builds a search engine to query them, and produces an interactive network map visualizing character relationships — exported from Gephi and rendered in-browser via Sigma.js.

---

## Folder Structure

```
GOT_relation_map/
├── awoiaf_pages/           ← HTML files from AWOIAF wiki pages
│   ├── A_certain_man.html
│   ├── Abelar_Hightower.html
│   └── ... (many more)
├── data/                   ← Processed data outputs
│   ├── got_characters.csv
│   ├── got_characters.pkl
│   ├── got_search_engine_df.pkl
│   └── GOT-characters-phrases-bert.csv
├── books/                  ← Raw .txt book files (GOT1.txt … GOT5.txt)
├── Gephi_files/            ← .gephi project files and GEXF exports
├── network-html-server/                   ← Sigma.js website (interactive network map)
│   ├── index.html
│   ├── network/            ← Graph data exported from Gephi
│   └── ...
├── 1_pipeline_per_chapter.ipynb
├── 2_scrap_characters.ipynb
├── 3_GOT-NER-model-compare.ipynb
├── 4_features_parser_&_search_engine.ipynb
├── 5_final_network_map.ipynb
└── README.md
```

---

## 1. Install Dependencies

```bash
pip install beautifulsoup4 pandas numpy networkx matplotlib scikit-learn scipy
pip install transformers torch tqdm spacy cloudscraper html5lib
```

---

## 2. Prepare Data

- Place the AWOIAF HTML files in the `awoiaf_pages/` folder.
- Place the raw book text files (`GOT1.txt` … `GOT5.txt`) in the `books/` folder.

---

## 3. Run the Pipeline

Execute the notebooks in order:

1. **1_pipeline_per_chapter.ipynb** — Processes raw book text chapter by chapter, runs NER with `dslim/bert-large-NER`, builds a character co-occurrence graph per chapter, and exports graph data.
2. **2_scrap_characters.ipynb** — Scrapes character names and wiki URLs from AWOIAF's character list, and collects aliases via infobox parsing.
3. **3_GOT-NER-model-compare.ipynb** — Compares NER approaches (spaCy `en_core_web_trf`, `fr_core_news_lg`, Hugging Face BERT-large-NER) using bag-of-words baselines and entity extraction benchmarks.
4. **4_features_parser_&_search_engine.ipynb** — Parses AWOIAF HTML pages, extracts 46 features across 11 families, builds a network graph for ranking (PageRank, HITS, centrality), and implements the three-stage search engine.
5. **5_final_network_map.ipynb** — Resolves raw NER entities against the canonical character database using the hybrid search engine, filters non-characters by score threshold, and outputs the final relationship graph for Gephi.

---

## How the Project Works

### Data Source and Parsing

HTML pages from AWOIAF are used rather than raw book text because they provide semi-structured, curated character information — infoboxes, categories, cross-references — enabling richer feature extraction. The `WikiHTMLParser` class uses BeautifulSoup to extract **46 distinct features** grouped into **11 families**:

- **Identity** — title, meta descriptions, canonical URLs
- **Names** — aliases, titles, name variants
- **Infobox** — allegiances, culture, family
- **Books** — appearances and roles per book
- **Text** — full article content and statistics
- **Structure** — sections, subsections, quotes
- **Links** — outgoing links and anchor texts
- **Categories** — wiki categories
- **Images** — portraits and galleries
- **Navboxes** — navigation elements
- **Page Size** — file size metrics

### NER and Entity Resolution

Three NER approaches are compared in Notebook 3 — spaCy (`en_core_web_trf`), Hugging Face BERT-large-NER, and a bag-of-words baseline. The best-performing model's raw entity extractions are then resolved in Notebook 5 against the canonical character database using the hybrid search engine, which scores and filters candidates to retain only true characters.

### Data Processing and Cleaning

Low-signal columns (redundant titles, empty fields) are dropped. Only characters with confirmed book appearances are retained. Numeric features are min-max normalized so that features with large scales (e.g., word count) do not dominate smaller ones (e.g., centrality scores) in distance computations.

### Network Analysis for Ranking

A directed graph is built where nodes are characters and edges are outgoing internal wiki links, mirroring the structure of the AWOIAF "web." Metrics computed:

- **PageRank** — overall importance via incoming links
- **HITS (Authority / Hub)** — distinguishes influential characters from connectors
- **Betweenness / Closeness Centrality** — identifies bridge characters and information-flow hubs

These metrics provide a data-driven ranking signal inspired by link-structure approaches used in web search engines.

### Dimensionality Reduction and Distance Metrics

PCA is applied to numeric features to identify principal axes of character variance. Euclidean and Minkowski distances from the origin surface characters that are particularly extreme or unique across the feature space, providing a novelty signal for search ranking.

### Book Role Parsing

Raw book strings (e.g., `"A Game of Thrones (POV)"`) are parsed into structured dictionaries with roles: `pov`, `appears`, `mentioned`, `appendix`. This enables context-aware search — POV characters receive a score boost of ×2.0 in book-scoped queries, while `mentioned` characters receive ×0.2.

### Search Engine Design

A three-stage pipeline:

1. **Context Filter** — optionally restrict to characters present in a specific book
2. **Retrieval** — find candidates via exact match, Jaccard similarity (set overlap), or TF-IDF cosine similarity (semantic matching)
3. **Ranking** — score candidates using hybrid mode (combines TF-IDF, network boosts, context roles) or specific metrics such as PageRank, centrality, or distance

### Character Relationship Network

The final co-occurrence graph (from Notebook 5) is exported as a GEXF file and loaded into **Gephi** for layout and visual styling. The network is then exported using Gephi's **Sigma.js plugin**, generating a self-contained website that runs entirely in the browser — no server required.

---

## 4. Interactive Network Map (Sigma.js)

The character relationship network is published as an interactive website built with **Gephi + Sigma.js**.

### How it was built

1. The co-occurrence graph produced by Notebook 5 was imported into **Gephi**.
2. A force-directed layout (ForceAtlas2) was applied to position nodes by relationship density.
3. Node size was mapped to PageRank score; color was mapped to community (modularity clustering).
4. The graph was exported using Gephi's **Sigma.js Export** plugin, which generates a ready-to-use `index.html` + `network/` data folder.

### How to view it

Open the exported site locally:

```bash
cd network-html-server/
python -m http.server 8080
# then open http://localhost:8080 in your browser
```

Or simply open `network-html-server/index.html` directly in a modern browser (Chrome/Firefox) — no server is needed for most browsers if CORS is not an issue.

### Features of the map

- **Zoom & pan** — scroll to zoom, click-drag to pan
- **Node search** — type a character name to highlight them and their connections
- **Hover tooltips** — shows character name and key attributes on hover
- **Click to focus** — clicking a node highlights its direct neighbors and fades the rest
- **Community coloring** — groups of strongly connected characters share a color

---

## 5. Outputs

| File | Description |
|---|---|
| `data/got_characters.csv` | Human-readable character feature table |
| `data/got_characters.pkl` | Full feature DataFrame (preserves complex objects) |
| `data/got_search_engine_df.pkl` | Enriched search index with network metrics |
| `data/GOT-characters-phrases-bert.csv` | Raw NER extractions from BERT |
| `Gephi_files/` | Gephi project and GEXF graph exports |
| `maps/` | Interactive Sigma.js network website |

---

## 6. Search Engine Usage

The search engine in `4_features_parser_&_search_engine.ipynb` supports querying characters with multiple ranking strategies:

```python
advanced_search_dynamic(
    query_name="Arya",
    df=master_df,
    current_book="A Game of Thrones",   # optional book filter
    retrieve_by="tfidf",                # "exact", "jaccard", or "tfidf"
    rank_by="hybrid",                   # "hybrid", "pagerank", "centrality", "distance"
    top_n=5
)
```

---

## Tuning Parameters

| Parameter | Location | Effect |
|---|---|---|
| `MAX_TOKENS_P` / `STRIDE_P` | Notebook 1 | NER chunk size and overlap |
| `COOC_WINDOW` | Notebook 1 | Co-occurrence window (sentences) |
| `ROLE_BOOSTS` | Notebook 5 | Score multipliers per book role |
| `top_n` | Notebook 4 & 5 | Number of search results returned |
| Gephi ForceAtlas2 settings | Gephi UI | Network layout density and spread |