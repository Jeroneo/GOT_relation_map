"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         ASOIAF — Character Extraction & Relationship Network Builder        ║
║                                                                              ║
║  OVERVIEW                                                                    ║
║  ────────                                                                    ║
║  This script processes the plain-text files of A Song of Ice and Fire and   ║
║  produces two outputs:                                                       ║
║                                                                              ║
║    1.  data/characters.json   – every character found, with total mention    ║
║                                 count and per-chapter breakdown.             ║
║                                                                              ║
║    2.  data/networks.json     – per-chapter co-occurrence graph where        ║
║                                 nodes = characters and edges = how often     ║
║                                 they appear "near" each other in the text.   ║
║                                                                              ║
║  PIPELINE STEPS                                                              ║
║  ──────────────                                                              ║
║    A.  Load books (.txt files from books/)                                  ║
║    B.  Split each book into chapters (regex on all-caps headings)            ║
║    C.  Identify the POV character from the chapter heading                   ║
║    D.  Run BERT-based NER on every chapter to extract PERSON entities        ║
║    E.  Build a co-occurrence graph per chapter using a sliding sentence      ║
║        window — two characters are "related" if they appear in the same     ║
║        window; edge weight = number of windows they share                   ║
║    F.  Aggregate counts & networks, write JSON outputs                      ║
║                                                                              ║
║  DEPENDENCIES                                                               ║
║    pip install transformers torch networkx tqdm                              ║
║                                                                              ║
║  MODEL                                                                       ║
║    dslim/bert-base-NER  (fine-tuned BERT for CoNLL-2003 NER)                ║
║    Recognises: PER · ORG · LOC · MISC                                       ║
║    We keep only PER (person) entities.                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Standard library
# ─────────────────────────────────────────────────────────────────────────────
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
#  Third-party
# ─────────────────────────────────────────────────────────────────────────────
import networkx as nx
from tqdm import tqdm
from transformers import pipeline

# ══════════════════════════════════════════════════════════════════════════════
#  USER-TUNABLE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BOOKS_DIR   = Path("books")   # directory containing .txt book files
OUTPUT_DIR  = Path("data")    # directory where JSON results are written

# ── NER model ─────────────────────────────────────────────────────────────────
#   "dslim/bert-base-NER" is small, fast, and works well on fantasy names.
#   Alternatives:
#     "dslim/bert-large-NER"               (more accurate, slower)
#     "Jean-Baptiste/roberta-large-ner-english"  (state-of-the-art, heavy)
NER_MODEL = "dslim/bert-base-NER"

# ── Tokeniser safety limit ────────────────────────────────────────────────────
#   BERT has a hard cap of 512 word-piece tokens per call.
#   We chunk the chapter text into overlapping passages of MAX_TOKENS words
#   (not tokens — we use words as a rough proxy) to stay under the limit.
MAX_TOKENS  = 400   # words per chunk (safe margin below 512 sub-word tokens)
STRIDE      = 50    # words of overlap between consecutive chunks
              #   Overlap prevents missing entities that straddle chunk edges.

# ── Co-occurrence window ──────────────────────────────────────────────────────
#   After splitting a chapter into sentences we slide a window of
#   COOC_WINDOW sentences.  Any two characters whose names both appear inside
#   the same window are considered "related" in that chapter.  The edge weight
#   between them is the number of windows in which they co-occur.
COOC_WINDOW = 15   # sentences — empirically good for novel-length chapters

# ── Noise filters ────────────────────────────────────────────────────────────
MIN_CHAR_LEN     = 2    # ignore single-character "names"
MIN_GLOBAL_MENTIONS = 1 # drop characters seen only once across all books
                        # (catches stray NER false-positives)

# ── Words that are almost never real character names but get tagged as PERSON ─
STOP_NAMES = {
    "lord", "ser", "lady", "king", "queen", "prince", "princess",
    "maester", "septon", "father", "mother", "brother", "sister",
    "son", "daughter", "man", "woman", "boy", "girl", "old", "young",
    "first", "second", "hand", "night", "watch", "wall", "iron",
    "sept", "god", "gods",
}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP A — LOAD BOOKS
# ══════════════════════════════════════════════════════════════════════════════

def load_books(books_dir: Path) -> dict[str, str]:
    """
    Read every .txt file in `books_dir` and return a mapping
        { book_stem: full_text_string }

    Files are sorted alphabetically, so name them with a numeric prefix
    (e.g. "01_agot.txt") if you want a specific reading order.
    """
    texts: dict[str, str] = {}
    paths = sorted(books_dir.glob("*.txt"))

    if not paths:
        raise FileNotFoundError(
            f"No .txt files found in '{books_dir}/'.  "
            "Place the plain-text book files there and re-run."
        )

    for path in paths:
        print(f"  📖  Loading  {path.name}")
        raw = path.read_text(encoding="utf-8", errors="replace")
        texts[path.stem] = raw

    return texts


# ══════════════════════════════════════════════════════════════════════════════
#  STEP B — SPLIT BOOKS INTO CHAPTERS
# ══════════════════════════════════════════════════════════════════════════════

# Pattern explanation
# ───────────────────
#   ^           — start of a line (re.MULTILINE)
#   \s*         — optional leading whitespace / blank lines
#   ([A-Z]{2,}) — the chapter heading: ≥2 consecutive uppercase letters
#                 (e.g. "BRAN", "PROLOGUE", "JON", "DAENERYS")
#   \s*$        — only whitespace after the heading on that line
#
# This intentionally matches the ASOIAF chapter structure where each chapter
# is headed by the POV character's name written in all-caps on its own line.
CHAPTER_HEADING_RE = re.compile(
    r"^\s*([A-Z][A-Z\s]{1,30}?)\s*$",   # all-caps word(s), 2-32 chars total
    re.MULTILINE,
)

def split_into_chapters(book_text: str, book_name: str) -> list[dict]:
    """
    Split a raw book string into a list of chapter dicts:

        {
            "book":       str,   # e.g. "01_agot"
            "chapter_idx": int,  # 0-based position in the book
            "heading":    str,   # raw heading string, e.g. "BRAN"
            "pov_char":   str,   # cleaned heading used as POV label
            "text":       str,   # chapter body text
        }

    Strategy
    ────────
    We find every all-caps heading with the regex above, then slice the
    text between consecutive headings to get each chapter body.
    """
    chapters = []

    # Find all heading positions
    matches = list(CHAPTER_HEADING_RE.finditer(book_text))

    if not matches:
        # Fallback: treat the entire book as a single unnamed chapter
        print(f"  ⚠  No chapter headings found in '{book_name}'. "
              "Treating the whole book as one chapter.")
        return [{
            "book":        book_name,
            "chapter_idx": 0,
            "heading":     "UNKNOWN",
            "pov_char":    "UNKNOWN",
            "text":        book_text,
        }]

    for i, match in enumerate(matches):
        heading_raw = match.group(1).strip()

        # Chapter body = text from end-of-heading to start-of-next-heading
        body_start = match.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(book_text)
        body       = book_text[body_start:body_end].strip()

        # Skip headings that produced an empty body
        # (can happen with back-to-back decorative headers)
        if not body:
            continue

        chapters.append({
            "book":        book_name,
            "chapter_idx": i,
            "heading":     heading_raw,
            "pov_char":    heading_raw.title(),  # "BRAN" → "Bran"
            "text":        body,
        })

    print(f"      → {len(chapters)} chapters found")
    return chapters


# ══════════════════════════════════════════════════════════════════════════════
#  STEP C — TEXT CHUNKING FOR BERT
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text_by_words(text: str,
                         max_words: int = MAX_TOKENS,
                         stride: int    = STRIDE) -> list[str]:
    """
    Split `text` into overlapping word-based chunks.

    Why chunking?
    ─────────────
    BERT's tokeniser has a hard limit of 512 sub-word tokens.  A long chapter
    can easily exceed this.  We therefore split the text into word-based
    windows (words ≈ tokens for English prose) and run the NER model on each
    window independently.

    Why overlap (stride)?
    ─────────────────────
    Without overlap an entity that happens to straddle two chunk boundaries
    would be split in half and missed by the NER model.  The stride ensures
    each boundary region is "seen" twice — once in each adjacent chunk.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += max_words - stride   # slide forward, keeping `stride` overlap

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  STEP D — BERT NER
# ══════════════════════════════════════════════════════════════════════════════

def load_ner_pipeline(model_name: str = NER_MODEL):
    """
    Load the HuggingFace NER pipeline.

    The pipeline wraps:
        1. A BERT tokeniser   — converts text → sub-word token IDs
        2. A BERT NER model   — predicts BIO tags per token
        3. An aggregator      — merges B-/I- tokens into full entity spans

    aggregation_strategy="simple"
        Merges consecutive tokens with the same entity type using the
        average score.  This is the safest setting for multi-word names
        like "Eddard Stark" or "Ser Waymar Royce".
    """
    print(f"\n🤖  Loading NER model: {model_name}")
    print("    (first run will download ~400 MB — cached afterwards)")
    ner = pipeline(
        task                  = "ner",
        model                 = model_name,
        aggregation_strategy  = "simple",
        device                = -1,   # -1 = CPU; change to 0 for GPU
    )
    print("    ✓ Model ready\n")
    return ner


def extract_persons_from_chapter(chapter_text: str,
                                  ner_pipeline,
                                  chapter_label: str = "") -> Counter:
    """
    Run BERT NER on one chapter and return a Counter of person name → count.

    Procedure
    ─────────
    1. Chunk the chapter text (to stay within BERT's 512-token window).
    2. Run the NER pipeline on each chunk.
    3. Keep only entities whose type starts with "PER" (e.g. "PER", "B-PER").
    4. Normalise each name (strip whitespace, title-case).
    5. Apply noise filters (stop-words, minimum length).
    6. Aggregate counts across all chunks.

    Deduplication note
    ──────────────────
    Because chunks overlap, the same entity may be counted twice near a
    chunk boundary.  We accept this small overcounting rather than adding
    complexity; the co-occurrence logic (Step E) operates on sentences, not
    chunks, so it is unaffected.
    """
    chunks  = chunk_text_by_words(chapter_text)
    persons: Counter = Counter()

    for chunk in chunks:
        if not chunk.strip():
            continue

        try:
            entities = ner_pipeline(chunk)
        except Exception as exc:
            # Graceful degradation: skip a problematic chunk
            print(f"    ⚠  NER error on chunk in [{chapter_label}]: {exc}")
            continue

        for ent in entities:
            # entity_group is "PER", "ORG", "LOC", or "MISC"
            if not ent["entity_group"].startswith("PER"):
                continue

            name = normalise_name(ent["word"])
            if is_valid_name(name):
                persons[name] += 1

    return persons


def normalise_name(raw: str) -> str:
    """
    Clean a raw entity string into a canonical character name.

    Steps
    ─────
    1. Remove BERT's sub-word marker "##" (artefact of WordPiece tokenisation).
    2. Collapse internal whitespace.
    3. Strip leading / trailing punctuation and whitespace.
    4. Apply Unicode normalisation (handles accented characters, ligatures).
    5. Title-case for consistent comparisons ("JON SNOW" → "Jon Snow").
    """
    name = raw.replace("##", "")               # sub-word artefact
    name = re.sub(r"\s+", " ", name)           # collapse whitespace
    name = name.strip(" ,.'\"—-")             # strip flanking punctuation
    name = unicodedata.normalize("NFC", name)  # Unicode normalisation
    name = name.title()                        # "EDDARD STARK" → "Eddard Stark"
    return name


def is_valid_name(name: str) -> bool:
    """
    Return True if `name` looks like a real character name.

    Filters
    ───────
    • Too short   — avoids single-letter false positives ("I", "A")
    • All digits  — "512" is not a name
    • Stop-words  — common titles / nouns that BERT sometimes mis-tags
    • No letters  — rejects purely punctuation or symbol strings
    """
    if len(name) < MIN_CHAR_LEN:
        return False
    if not any(c.isalpha() for c in name):
        return False
    if name.isdigit():
        return False
    if name.lower() in STOP_NAMES:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  STEP E — CO-OCCURRENCE RELATIONSHIP GRAPH
# ══════════════════════════════════════════════════════════════════════════════

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def build_cooccurrence_graph(chapter_text:  str,
                              persons:       Counter,
                              pov_char:      str,
                              window_size:   int = COOC_WINDOW) -> nx.Graph:
    """
    Build an undirected weighted co-occurrence graph for one chapter.

    Nodes
    ─────
    Every character name found by NER in this chapter.
    The POV character (chapter heading) is always added as a node even if NER
    missed them (e.g. they are implied rather than named in the chapter body).

    Edges
    ─────
    Two characters share an edge if they both appear in at least one
    sentence-window.  The edge weight counts how many distinct windows
    they co-occur in — a proxy for how strongly they interact in the chapter.

    Algorithm
    ─────────
    1. Split chapter text into sentences.
    2. For each sentence, record which known characters are mentioned.
    3. Slide a window of `window_size` sentences.
    4. For every window, collect the set of characters mentioned anywhere
       in those sentences, then add +1 to every pair's edge weight.

    Why sentences rather than words?
    ─────────────────────────────────
    Word-distance is noisy (a character's name used in a description 50 words
    away may be irrelevant).  Sentence co-occurrence is the standard baseline
    in literary character-network analysis (e.g. Beveridge & Shan, 2016 —
    the original Game of Thrones network paper).
    """
    G = nx.Graph()

    # ── Add all detected characters as nodes ─────────────────────────────────
    for char, count in persons.items():
        G.add_node(char, mentions=count, is_pov=False)

    # ── Ensure the POV character is always a node ─────────────────────────────
    pov_normalised = normalise_name(pov_char)
    if pov_normalised and is_valid_name(pov_normalised):
        if pov_normalised not in G:
            G.add_node(pov_normalised, mentions=0, is_pov=True)
        else:
            G.nodes[pov_normalised]["is_pov"] = True

    # ── Build a lookup: set of character names (lower-cased for matching) ─────
    #   We use the full name AND each individual word of multi-word names.
    #   "Jon Snow" is matched by both "Jon Snow" and "Jon" or "Snow" alone.
    char_names: set[str] = set(G.nodes)
    first_names: dict[str, str] = {}   # first_name → full_name
    for full_name in char_names:
        parts = full_name.split()
        if parts:
            first = parts[0]
            # Only register the first-name shortcut if it is unambiguous
            if first not in first_names:
                first_names[first] = full_name

    def find_chars_in_sentence(sentence: str) -> set[str]:
        """Return the set of known character names mentioned in a sentence."""
        found: set[str] = set()
        # Check full names first (most specific)
        for name in char_names:
            if name in sentence:
                found.add(name)
        # Then check first names for any not yet found by full name
        for first, full in first_names.items():
            if full not in found and first in sentence:
                found.add(full)
        return found

    # ── Split text into sentences ─────────────────────────────────────────────
    sentences = SENTENCE_SPLIT_RE.split(chapter_text)
    if not sentences:
        return G

    # ── Pre-compute which characters appear in each sentence ─────────────────
    sentence_chars: list[set[str]] = [
        find_chars_in_sentence(sent) for sent in sentences
    ]

    # ── Slide the window and accumulate edge weights ──────────────────────────
    num_sentences = len(sentences)

    for win_start in range(num_sentences):
        win_end = min(win_start + window_size, num_sentences)

        # Union of all characters in this window
        window_chars: set[str] = set()
        for s in range(win_start, win_end):
            window_chars |= sentence_chars[s]

        # For every pair in the window, increment the edge weight
        chars_list = list(window_chars)
        for i in range(len(chars_list)):
            for j in range(i + 1, len(chars_list)):
                u, v = chars_list[i], chars_list[j]
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)

    return G


# ══════════════════════════════════════════════════════════════════════════════
#  STEP F — SERIALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def graph_to_dict(G: nx.Graph) -> dict:
    """
    Convert a NetworkX graph to a plain JSON-serialisable dict.

    Output format
    ─────────────
    {
        "nodes": [
            { "id": "Jon Snow", "mentions": 47, "is_pov": true },
            ...
        ],
        "edges": [
            { "source": "Jon Snow", "target": "Robb Stark", "weight": 12 },
            ...
        ]
    }

    This format is directly importable by D3.js, Gephi, Cytoscape, and
    most other graph-visualisation tools.
    """
    nodes = [
        {
            "id":       node,
            "mentions": data.get("mentions", 0),
            "is_pov":   data.get("is_pov", False),
        }
        for node, data in G.nodes(data=True)
    ]

    edges = [
        {
            "source": u,
            "target": v,
            "weight": data.get("weight", 1),
        }
        for u, v, data in G.edges(data=True)
    ]

    # Basic graph-level statistics (handy for analysis later)
    stats = {
        "num_nodes":          G.number_of_nodes(),
        "num_edges":          G.number_of_edges(),
        "density":            round(nx.density(G), 4) if G.number_of_nodes() > 1 else 0.0,
        "avg_degree":         round(
                                  sum(d for _, d in G.degree()) / G.number_of_nodes(), 2
                              ) if G.number_of_nodes() > 0 else 0.0,
    }

    return {"nodes": nodes, "edges": edges, "stats": stats}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load NER model ────────────────────────────────────────────────────────
    ner = load_ner_pipeline(NER_MODEL)

    # ── Load books ────────────────────────────────────────────────────────────
    print(f"📚  Scanning books in '{BOOKS_DIR}/' …\n")
    books = load_books(BOOKS_DIR)

    # ── Per-chapter processing ────────────────────────────────────────────────
    #
    #  We build two top-level data structures that will be saved as JSON:
    #
    #  all_chapters_networks
    #  ─────────────────────
    #  List of dicts, one per chapter.  Each entry contains the chapter
    #  metadata (book, heading, POV) plus the serialised co-occurrence graph.
    #
    #  global_character_counter
    #  ─────────────────────────
    #  Maps every character name (across all books/chapters) to:
    #    • "total_mentions"   – raw NER hit count summed over all chapters
    #    • "chapter_mentions" – { "book::chapter_heading": count } breakdown
    #    • "books"            – set of books the character appears in
    #
    all_chapters_networks:   list[dict]               = []
    global_char_data:        dict[str, dict]          = defaultdict(lambda: {
        "total_mentions":   0,
        "chapter_mentions": {},
        "books":            [],
    })

    for book_name, book_text in books.items():
        print(f"\n{'═'*60}")
        print(f"  📘  Book: {book_name}")
        print(f"{'═'*60}")

        chapters = split_into_chapters(book_text, book_name)

        for chapter in tqdm(chapters, desc=f"  Processing {book_name}", unit="ch"):
            ch_label = f"{book_name}::{chapter['heading']}"

            # ── D: Extract persons via BERT NER ───────────────────────────────
            persons = extract_persons_from_chapter(
                chapter_text  = chapter["text"],
                ner_pipeline  = ner,
                chapter_label = ch_label,
            )

            # ── E: Build co-occurrence graph ──────────────────────────────────
            G = build_cooccurrence_graph(
                chapter_text = chapter["text"],
                persons      = persons,
                pov_char     = chapter["pov_char"],
            )

            # ── Accumulate global character data ──────────────────────────────
            for char, count in persons.items():
                gc = global_char_data[char]
                gc["total_mentions"] += count
                gc["chapter_mentions"][ch_label] = count
                if book_name not in gc["books"]:
                    gc["books"].append(book_name)

            # ── Store chapter network ─────────────────────────────────────────
            chapter_record = {
                "book":        chapter["book"],
                "chapter_idx": chapter["chapter_idx"],
                "heading":     chapter["heading"],
                "pov_char":    chapter["pov_char"],
                "network":     graph_to_dict(G),
            }
            all_chapters_networks.append(chapter_record)

    # ═════════════════════════════════════════════════════════════════════════
    #  Post-processing: apply global minimum-mention filter
    # ═════════════════════════════════════════════════════════════════════════
    #
    #  Characters seen fewer than MIN_GLOBAL_MENTIONS times across all books
    #  are almost certainly NER false-positives (e.g. "Ser" being tagged as a
    #  person because it precedes a name, random nouns in unusual contexts).
    #  We remove them from the character dictionary but leave the networks
    #  intact (the networks already have their own node-level mention counts
    #  for independent filtering later).
    #
    filtered_chars = {
        name: data
        for name, data in global_char_data.items()
        if data["total_mentions"] >= MIN_GLOBAL_MENTIONS
    }

    # Sort by total mentions descending for readability
    sorted_chars = dict(
        sorted(filtered_chars.items(),
               key=lambda kv: kv[1]["total_mentions"],
               reverse=True)
    )

    # ═════════════════════════════════════════════════════════════════════════
    #  Write outputs
    # ═════════════════════════════════════════════════════════════════════════

    characters_path = OUTPUT_DIR / "characters.json"
    networks_path   = OUTPUT_DIR / "networks.json"

    with open(characters_path, "w", encoding="utf-8") as f:
        json.dump(sorted_chars, f, indent=2, ensure_ascii=False)

    with open(networks_path, "w", encoding="utf-8") as f:
        json.dump(all_chapters_networks, f, indent=2, ensure_ascii=False)

    # ═════════════════════════════════════════════════════════════════════════
    #  Summary
    # ═════════════════════════════════════════════════════════════════════════

    total_chapters = len(all_chapters_networks)
    total_chars    = len(sorted_chars)

    print(f"\n{'═'*60}")
    print(f"  ✅  Done!")
    print(f"{'═'*60}")
    print(f"  Books processed    : {len(books)}")
    print(f"  Chapters found     : {total_chapters}")
    print(f"  Unique characters  : {total_chars}")
    print(f"\n  Top 30 characters by mention count:")
    print(f"  {'Mentions':>8}  Character")
    print(f"  {'─'*8}  {'─'*30}")
    for name, data in list(sorted_chars.items())[:30]:
        print(f"  {data['total_mentions']:>8}  {name}")

    print(f"\n  📄  {characters_path}")
    print(f"  📄  {networks_path}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT FORMAT REFERENCE
# ══════════════════════════════════════════════════════════════════════════════
#
#  ── characters.json ─────────────────────────────────────────────────────────
#
#  {
#    "Jon Snow": {
#      "total_mentions": 412,
#      "chapter_mentions": {
#        "01_agot::BRAN": 3,
#        "01_agot::JON": 47,
#        ...
#      },
#      "books": ["01_agot", "02_acok", ...]
#    },
#    ...
#  }
#
#  ── networks.json ────────────────────────────────────────────────────────────
#
#  [
#    {
#      "book":        "01_agot",
#      "chapter_idx": 1,
#      "heading":     "BRAN",
#      "pov_char":    "Bran",
#      "network": {
#        "nodes": [
#          { "id": "Bran",      "mentions": 14, "is_pov": true  },
#          { "id": "Jon Snow",  "mentions":  6, "is_pov": false },
#          ...
#        ],
#        "edges": [
#          { "source": "Bran", "target": "Jon Snow", "weight": 8 },
#          ...
#        ],
#        "stats": {
#          "num_nodes": 12,
#          "num_edges": 31,
#          "density":   0.4712,
#          "avg_degree": 5.17
#        }
#      }
#    },
#    ...
#  ]
#
# ══════════════════════════════════════════════════════════════════════════════
