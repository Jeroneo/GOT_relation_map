# GOT Character Feature Extractor and Search Engine

This project parses HTML pages from A Wiki of Ice and Fire (AWOIAF) to extract features from Game of Thrones characters and builds a search engine for querying them.

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
│   └── got_search_engine_df.pkl
├── books/                  ← (if applicable)
├── Gephi_files/            ← Graph files for visualization
├── maps/                   ← Map-related files
├── 1.pipeline_per_chapter.ipynb     ← Pipeline per chapter
├── 2.scrap_characters.ipynb         ← Character scraping
├── 3.GOT-NER-model-compare.ipynb    ← NER model comparison
├── 4.features_parser_&_search_engine.ipynb  ← Main feature extraction and search engine
└── README.md
```

## 1. Install Dependencies

```bash
pip install beautifulsoup4 pandas numpy re os glob ast warnings networkx matplotlib sklearn scipy
```

## 2. Prepare Data

Place the AWOIAF HTML files in the `awoiaf_pages/` folder.

## 3. Run the Pipeline

Execute the notebooks in order:

1. **1.pipeline_per_chapter.ipynb**: Processes data per chapter.
2. **2.scrap_characters.ipynb**: Scrapes character information.
3. **3.GOT-NER-model-compare.ipynb**: Compares NER models.
4. **4.features_parser_&_search_engine.ipynb**: Extracts features from HTML, builds search engine.

Run each notebook in a Jupyter environment.

## How the Project Works

This project leverages structured data from A Wiki of Ice and Fire (AWOIAF) wiki pages to create a comprehensive feature set for Game of Thrones characters and enable advanced search capabilities. Here's a breakdown of the approach and design choices:

### Data Source and Parsing
- **Choice**: HTML pages from AWOIAF were chosen over raw book text because they provide semi-structured, curated information about characters, including infoboxes, categories, and cross-references. This allows for richer feature extraction compared to unstructured text.
- **Method**: The `WikiHTMLParser` class uses BeautifulSoup to parse HTML and extract 46 distinct features, grouped into 11 families:
  - **Identity**: Title, meta descriptions, canonical URLs.
  - **Names**: Aliases, titles, name variants.
  - **Infobox**: Biographical details like allegiances, culture, family.
  - **Books**: Appearances and roles in each book.
  - **Text**: Full article content and statistics.
  - **Structure**: Sections, subsections, quotes.
  - **Links**: Outgoing links and anchor texts.
  - **Categories**: Wiki categories.
  - **Images**: Portraits and galleries.
  - **Navboxes**: Navigation elements.
  - **Page Size**: File size metrics.
- **Why 46 features?**: To capture a holistic view of each character, enabling both semantic search (via text content) and structured queries (via metadata).

### Data Processing and Cleaning
- **Cleaning**: Low-signal columns (e.g., redundant titles, empty fields) are dropped. Only characters with book appearances are retained to focus on canonical characters. Numeric features are min-max normalized to ensure equal contribution in distance and ranking computations.
- **Choice**: Normalization prevents features with larger scales (e.g., word count) from dominating smaller ones (e.g., centrality scores).

### Network Analysis for Ranking
- **Graph Construction**: A directed graph is built where nodes are characters and edges are outgoing internal wiki links. This represents the "web" of relationships in the wiki.
- **Metrics Computed**:
  - **PageRank**: Measures overall importance based on incoming links.
  - **HITS (Authority/Hub)**: Distinguishes between influential characters (authorities) and connectors (hubs).
  - **Centrality (Betweenness/Closeness)**: Identifies bridge characters and those central to information flow.
- **Choice**: Network metrics provide a data-driven way to rank characters by prominence, complementing content-based features. This is inspired by web search engines like Google, where link structure indicates relevance.

### Dimensionality Reduction and Distance Metrics
- **PCA**: Applied to numeric features to identify principal axes of variance, helping understand which features drive character differences.
- **Distances**: Euclidean and Minkowski distances from the origin measure how "extreme" or unique a character is across features.
- **Choice**: These provide novelty signals for search ranking, surfacing characters that stand out in the feature space.

### Book Role Parsing
- **Method**: Raw book strings (e.g., "A Game of Thrones (POV)") are parsed into structured dictionaries with roles like 'pov', 'appears', 'mentioned', 'appendix'.
- **Choice**: Enables context-aware search, boosting characters based on their role in specific books (e.g., POV characters get higher scores in book-scoped queries).

### Search Engine Design
- **Three-Stage Pipeline**:
  1. **Context Filter**: Optionally restrict to characters in a specific book.
  2. **Retrieval**: Find candidates via exact match, Jaccard similarity (set overlap), or TF-IDF cosine similarity (semantic matching).
  3. **Ranking**: Score candidates using hybrid (combines TF-IDF, network boosts, context roles), or specific metrics like PageRank, centrality, or distances.
- **Choices**:
  - **Multi-method retrieval**: Supports both keyword (exact/Jaccard) and semantic (TF-IDF) queries.
  - **Flexible ranking**: Hybrid mode balances content relevance, network importance, and context. Distance metrics sort ascending (closer to origin first) for similarity-based ranking.
  - **Why not a simple database?**: The engine handles complex, multi-dimensional queries with custom scoring, outperforming basic lookups for exploratory search.

### Implementation Choices
- **Notebooks**: Used for interactive development, allowing step-by-step execution, visualization, and easy parameter tuning. This is ideal for research and iteration compared to scripts.
- **Libraries**: Standard Python stack (BeautifulSoup for parsing, pandas for data, NetworkX for graphs, scikit-learn for TF-IDF/PCA) chosen for reliability, performance, and ecosystem compatibility.
- **Outputs**: Data saved as CSV (human-readable), pickle (preserves complex objects), and search index (optimized for querying).

This design transforms static wiki pages into a dynamic, queryable knowledge base, enabling users to discover characters through semantic, structural, and contextual lenses.

## 4. Outputs

- Processed character data in `data/got_characters.csv` and `data/got_characters.pkl`.
- Search index in `data/got_search_engine_df.pkl`.

## 5. Search Engine Usage

The search engine in `4.features_parser_&_search_engine.ipynb` allows querying characters with various ranking methods, including hybrid, TF-IDF, PageRank, etc.

## Tuning Parameters

Adjust parameters in the notebooks as needed, such as thresholds for feature extraction, similarity measures, and ranking metrics.
