# 🌾 Kisan Sahayak

**A multilingual Government Scheme and Farmer Advisory RAG assistant for Marathi, Hindi, and English.**

Kisan Sahayak searches a local collection of official government and agricultural PDF documents, retrieves the most relevant page-level excerpts, and asks a language model to answer **only from those excerpts**. Every answer is designed to be checked against its sources. When the documents do not contain enough evidence, the assistant should refuse rather than guess.

This project was built as a transparent, inspectable RAG application. It does **not** use LangChain, fine-tuning, agents, or multi-agent workflows.

---

## Table of contents

- [Quick start (after cloning)](#quick-start-after-cloning)

1. [Project status](#1-project-status)
2. [What the application does](#2-what-the-application-does)
3. [What was built in this project](#3-what-was-built-in-this-project)
4. [What is external or third-party](#4-what-is-external-or-third-party)
5. [Concepts explained simply](#5-concepts-explained-simply)
6. [Architecture](#6-architecture)
7. [Technology stack and why these choices were made](#7-technology-stack-and-why-these-choices-were-made)
8. [The corpus and data preparation](#8-the-corpus-and-data-preparation)
9. [Ingestion and indexing](#9-ingestion-and-indexing)
10. [Retrieval design](#10-retrieval-design)
11. [Answer generation, citations, and safety](#11-answer-generation-citations-and-safety)
12. [Streamlit user interface](#12-streamlit-user-interface)
13. [How to run the project](#13-how-to-run-the-project)
14. [How to test the project](#14-how-to-test-the-project)
15. [How to demonstrate it to your guide](#15-how-to-demonstrate-it-to-your-guide)
16. [What you can do with it right now](#16-what-you-can-do-with-it-right-now)
17. [Evaluation results](#17-evaluation-results)
18. [What is still remaining](#18-what-is-still-remaining)
19. [Known limitations](#19-known-limitations)
20. [Security and privacy](#20-security-and-privacy)
21. [Troubleshooting](#21-troubleshooting)
22. [Commands at a glance](#22-commands-at-a-glance)

---

## Quick start (after cloning)

Use this path when a teammate clones or pulls the repository onto a new machine.

> The repository intentionally excludes the PDF corpus, `.env`, `.venv/`, `storage/`, and `.local/`. You must restore the documents and create your own local configuration before building the index.

### 1. Clone and enter the project

```bash
git clone <your-private-repo-url>.git
cd kisan-sahayak
```

### 2. Create the Python 3.11 environment

Use Python **3.11**. If your system default is different, replace `python3.11` with the path to a 3.11 interpreter.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

### 3. Restore the PDF corpus

The 26 source PDFs are about 71 MB and are distributed separately from Git.

1. Get the team `data/raw` archive from the agreed shared drive or college storage.
2. Extract it at the project root so the folders look like this:

```text
kisan-sahayak/
└── data/
    └── raw/
        ├── central_schemes/
        ├── crop_advisory/
        ├── maharashtra_schemes/
        ├── soil_and_fertilizer/
        └── weather_advisory/
```

3. Do **not** rename or reorder the files. The inventory uses their original filenames and SHA-256 hashes. See `data/raw/README.md` for the full corpus-restoration notes.

### 4. Install Tesseract for OCR

Ingestion uses OCR for scanned/screenshot-heavy pages. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-mar tesseract-ocr-hin tesseract-ocr-eng

.venv/bin/python ingest.py --ocr-health
```

Leave `TESSERACT_CMD` blank in `.env` to use the system `tesseract` binary. If you install Tesseract somewhere custom, set `TESSERACT_CMD` to that absolute path.

### 5. Create your local `.env`

```bash
cp .env.example .env
```

Edit `.env` and add your own key. Leave `GEMINI_MODEL` empty for now:

```env
GEMINI_API_KEY=your-own-key-here
GEMINI_MODEL=
```

Then list the models available to your key and copy one printed model ID into `GEMINI_MODEL`:

```bash
.venv/bin/python ingest.py --list-models
```

Do not guess a model name and do not commit `.env`. Every teammate should use their own key.

### 6. Build the index and verify

```bash
.venv/bin/python ingest.py
.venv/bin/python ingest.py --stats
.venv/bin/python -m pytest -q
```

Expected results:

```text
ChromaDB collection count: 2240
59 passed
```

The first ingestion downloads the embedding model and builds ChromaDB, so it takes a few minutes. Later runs are incremental and skip unchanged documents.

### 7. Run the app

```bash
.venv/bin/streamlit run app.py
```

Open the URL printed by Streamlit, usually `http://localhost:8501`.

Without a key, the app still starts and retrieval/tests work, but it shows a configuration message instead of generating answers.

### Optional: use local Ollama instead of Gemini

Set these in `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

Then start the local server in another terminal:

```bash
ollama serve
ollama pull llama3.2:3b
```

Never commit `.env`, `data/raw/**/*.pdf`, `storage/`, `.venv/`, or `.local/`.

---

## 1. Project status

The project is **indexed, tested, and live-verified**. Retrieval works without any API key. With a configured Gemini key and model, all 24 answerable evaluation questions produced verified live answers and all six unsupported questions abstained.

| Item | Current result |
|---|---|
| Curated PDFs | **26 unique documents** |
| Inventory rows | **29 rows**: 26 included and 3 exact duplicates excluded |
| PDF pages | **851 unique pages** for the included documents |
| Extracted text | approximately **2.30 million characters** |
| Indexed chunks | **2,240** |
| OCR pages | **97** |
| Ingestion failures | **0** |
| ChromaDB collection | `farmer_knowledge` |
| Automated tests | **59 passed** |
| Dependency check | `pip check` reports no broken requirements |
| Retrieval evaluation | **100% hit@5**, **MRR 1.000**, **recall 0.979**, mean top score **0.7822** |
| Runtime similarity threshold | **0.66** (configured default and `.env.example` value) |
| Streamlit smoke test | **Passed**: no-key configuration message and configured startup |
| Incremental ingestion | **Passed**: 26 unchanged documents skipped; collection stayed at 2,240 |
| Live Gemini verification | **Completed**: all 24 answerable questions produced verified live answers; 6/6 unsupported questions abstained |
| `.env` in this workspace | **Present** (ignored by Git) |
| `GEMINI_API_KEY` in this workspace | **Present** in `.env`; never printed or committed |

> **Important distinction:** the 100% retrieval hit rate means the expected **document** was retrieved in the top 5 results. Live answer verification additionally checked generated Marathi, Hindi, and English answers, citations, abstention, and graceful API-failure behavior. It is not a formal human fact-check of every claim.

---

## 2. What the application does

A farmer or advisor can ask a question in Marathi, Hindi, or English, for example:

- “What is PM-KISAN and how much financial support does it provide?”
- “पीएम किसान योजनेसाठी कोण पात्र आहे?”
- “कापूस पिकातील एकात्मिक कीड व्यवस्थापनासाठी कोणत्या पद्धती सुचवल्या आहेत?”
- “How does a farmer register on the Maha-DBT portal?”

The application then:

1. Searches the local document index for relevant excerpts.
2. Builds a bounded context from the best excerpts.
3. Adds source IDs such as `[S1]` and `[S2]`.
4. Sends the excerpts and question to the configured language model.
5. Requires the model to answer from the excerpts only.
6. Validates the source markers in the answer.
7. Shows the answer with supporting sources and page numbers.
8. Returns a localized “not found” message when the evidence is insufficient.

The system supports:

- Central Government schemes
- Maharashtra State schemes
- Crop advisories and integrated pest management
- Soil and fertilizer information
- Weather and agromet advisories

It does **not** currently provide:

- Live weather forecasts
- Live market prices
- Live deadlines
- Personalized legal, financial, or eligibility decisions
- Verified official URLs where the source inventory does not contain a URL

---

## 3. What was built in this project

This section distinguishes the original project work from the third-party libraries and models it uses. This is useful when explaining the project to a guide or evaluator.

### Original implementation

The following parts are project-specific code written for Kisan Sahayak:

- **Document inventory and curation workflow**
  - Selecting 26 unique documents
  - Identifying and excluding 3 exact duplicates
  - Capturing categories, titles, filenames, page counts, hashes, language, dates, and validity status

- **Custom ingestion pipeline**
  - Document discovery
  - SHA-256 hashing
  - Incremental skip logic
  - Manifest creation and updates
  - Stable chunk IDs
  - Atomic replacement of stale chunks
  - Per-document failure reporting

- **PDF text extraction and OCR fallback logic**
  - Page-level extraction
  - Detection of low-text or garbled pages
  - Forced OCR for two known legacy-font Marathi documents
  - Tesseract detection, environment setup, and OCR health checks

- **Conservative cleaning layer**
  - Removal of repeated headers, footers, and page numbers
  - Preservation of actual document content

- **Custom chunking**
  - Page-aware sliding word windows
  - 200 words per window and 100 words of overlap
  - Page-start and page-end metadata
  - Title/scheme prefixes for embeddings

- **Local embedding integration**
  - Loading the multilingual Sentence Transformers model
  - Batched encoding
  - Normalized vectors
  - CPU execution
  - Offline/cached model operation

- **ChromaDB integration**
  - Persistent collection
  - Metadata filtering
  - Cosine-similarity queries
  - Insert, upsert, count, and stale-chunk removal

- **Multilingual retrieval improvements**
  - Marathi/Hindi scheme-name query expansion
  - Section-level intent hints for procedures and scheme benefits
  - Title-aware embeddings
  - Per-document diversification
  - Fill-to-`top_k` behavior
  - Similarity threshold support

- **RAG orchestration**
  - Retrieval
  - Context construction with source IDs
  - Prompt assembly
  - Citation validation
  - Abstention behavior
  - LLM error handling
  - Privacy redaction

- **Streamlit application**
  - Language selection
  - Chat interface
  - Suggested questions
  - Source cards
  - Configuration and health messages

- **Tests and evaluation**
  - 59 automated tests
  - A 30-question evaluation dataset
  - Retrieval evaluation with hit rate, MRR, and recall
  - Live generation evaluation scaffolding

### What was not trained from scratch

The project does **not** train or fine-tune an embedding model or an LLM. It uses:

- A pretrained multilingual embedding model
- A hosted Gemini model, or an optional local Ollama model
- A pretrained OCR engine

The contribution is the **data pipeline, retrieval design, grounding rules, citation handling, application, and evaluation** around those components.

---

## 4. What is external or third-party

| Component | Role | External? |
|---|---|---|
| PyMuPDF | Extracts text and page structure from PDFs | Yes |
| Tesseract OCR | Reads scanned or legacy-font pages | Yes |
| `paraphrase-multilingual-MiniLM-L12-v2` | Converts text into multilingual vectors | Yes, pretrained model |
| ChromaDB | Stores vectors and metadata; nearest-neighbour search | Yes |
| Gemini / `google-genai` | Generates natural-language answers | Yes, hosted service |
| Ollama | Optional local LLM runtime | Yes, optional |
| Streamlit | Browser UI | Yes |
| Pydantic Settings | Environment and configuration validation | Yes |
| pytest | Test runner | Yes |

The project code connects these pieces and adds the domain-specific logic, metadata, safety rules, retrieval tuning, UI, tests, and documentation.

---

## 5. Concepts explained simply

If you are presenting this to a guide, these explanations should let you describe the system without treating it as a black box.

### 5.1 What is RAG?

**RAG** means **Retrieval-Augmented Generation**.

A normal language model answers from its pretrained knowledge. That can be outdated, unsupported, or invented. RAG adds a retrieval step:

```text
Question
   ↓
Search a trusted document collection
   ↓
Retrieve relevant excerpts
   ↓
Give those excerpts to the language model
   ↓
Generate an answer grounded in the excerpts
   ↓
Show the supporting sources
```

The retrieval step is like giving the model an open-book exam. The model is not trusted to remember facts; it must work from the pages placed in front of it.

In this project:

- The “book” is the local PDF corpus.
- The “index” is ChromaDB.
- The “search” uses embeddings.
- The “answer” is produced by Gemini or Ollama.
- The citations let the user verify the answer.

### 5.2 What is an embedding?

An embedding is a list of numbers that represents the **meaning** of a piece of text.

For example, the model might represent:

- “PM-KISAN farmer income support”
- “पीएम किसान शेतकरी आर्थिक सहाय्य”

as two vectors that are close together, because they mean similar things even though the scripts are different.

This project uses:

- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Vector size: **384 dimensions**
- Vectors are normalized to unit length
- Similarity is measured with **cosine similarity**
- The model runs locally on CPU

An embedding is not a keyword index. It captures semantic similarity. That is why a Marathi question can retrieve an English page.

### 5.3 What is a vector database?

A normal database searches exact fields. A vector database searches by **similarity of meaning**.

ChromaDB stores:

- The vector for each chunk
- The original chunk text
- Metadata such as document ID, title, filename, page number, category, language, and validity status

When a question arrives:

1. The question is converted into a vector.
2. ChromaDB compares that vector with the stored chunk vectors.
3. It returns the chunks whose vectors are closest.

This project uses a persistent local Chroma collection named `farmer_knowledge`.

### 5.4 What is cosine similarity?

Cosine similarity measures the angle between two vectors:

- Close to `1.0`: very similar direction and meaning
- Close to `0.0`: little relationship
- Negative: opposite direction

The app stores a score based on Chroma’s cosine distance:

```text
similarity = 1 - cosine_distance
```

The retrieval evaluation uses these scores to choose a conservative cutoff.

### 5.5 What is chunking?

Embedding models cannot accept an entire PDF at once. The document must be split into smaller passages called **chunks**.

Chunk size is a trade-off:

- Very small chunks may lose context.
- Very large chunks may not fit the embedding model’s token limit.
- Overlap helps avoid losing information at chunk boundaries.

The embedding model has a **128-token maximum sequence length**. A long passage would be silently truncated, so part of it would never be represented in the index.

This project handles that with:

- **200-word windows**
- **100-word overlap**
- **Title and scheme prefix** in the embedding text
- Page-aware metadata

Conceptually:

```text
Window 1: words   1–200
Window 2: words 101–300
Window 3: words 201–400
```

The next window reuses the previous 100 words. Every part of a long document appears near the start of some indexed window, while the complete original chunk is still available to the LLM.

### 5.6 What is top-k retrieval?

`top_k` is the number of chunks returned by search. This project defaults to **5**.

The retriever:

1. Fetches extra candidates.
2. Applies the optional similarity threshold.
3. Prefers at most two chunks from any one document.
4. Fills the remaining slots if diversification makes the list too short.

The goal is to give the answer model several relevant, diverse excerpts without filling the context window with repeated text from one long manual.

### 5.7 What is a relevance threshold?

A threshold rejects weak matches.

- Too low: irrelevant excerpts reach the model.
- Too high: relevant excerpts are rejected and the app gives false refusals.

For the current index, the measured scores are:

- Lowest supported answerable question: **0.7200**
- Highest unsupported question: **0.6590**

A threshold of **0.66** separates those cases in the current evaluation set. It is a calibrated starting point, not a universal constant. Revisit it after adding documents or changing the embedding model.

### 5.8 What is grounding and abstention?

**Grounding** means every factual claim must be supported by retrieved text.

**Abstention** means the assistant says “I could not find sufficient verified information” instead of inventing an answer.

The system prompt tells the model to:

- Use only the excerpts
- Cite sources
- Refuse when evidence is insufficient
- Warn when a document is historical
- Never fabricate amounts, dates, URLs, or eligibility rules

### 5.9 What is a citation?

A citation connects a statement in the answer to a source excerpt.

The app gives the model sources like:

```text
[S1] PM-KISAN FAQ, page 1
[S2] PM-KISAN operational guidelines, page 2
```

The model is told to write claims such as:

```text
The scheme provides ₹6,000 per year in three instalments [S1].
```

The app then checks that `S1` and `S2` are real source IDs that were actually retrieved. Invalid IDs are removed.

---

## 6. Architecture

### 6.1 High-level diagram

```text
                         ┌─────────────────────────────┐
                         │ data/raw/ PDF documents     │
                         │ + document_inventory.csv    │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────┐
                    │ Ingestion pipeline                 │
                    │                                    │
                    │ 1. Hash and skip unchanged files  │
                    │ 2. Extract pages with PyMuPDF      │
                    │ 3. OCR low-text/garbled pages      │
                    │ 4. Clean headers/footers           │
                    │ 5. Build 200/100 sliding windows   │
                    │ 6. Add title/scheme to embedding   │
                    │ 7. Upsert vectors + metadata       │
                    │ 8. Update the manifest             │
                    └──────────────┬─────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────────┐
                    │ ChromaDB: farmer_knowledge         │
                    │ 2,240 chunks + page metadata       │
                    └──────────────┬─────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
          Offline retrieval evaluation      Streamlit chat UI
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────────┐
                    │ Question processing                │
                    │ 1. Safety/redaction                │
                    │ 2. Scheme-name query expansion     │
                    │ 3. Embedding + intent fusion       │
                    │ 4. Top-k search + threshold        │
                    │ 5. Per-document diversification    │
                    │ 6. Context + source IDs            │
                    └──────────────┬─────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────────┐
                    │ Gemini or Ollama                   │
                    │ strict grounded-answer prompt      │
                    └──────────────┬─────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────────┐
                    │ Answer validation                  │
                    │ - parse citations                  │
                    │ - remove invalid source IDs        │
                    │ - abstain when unsupported         │
                    │ - show page-level sources          │
                    └──────────────┬─────────────────────┘
                                   │
                                   ▼
                              Farmer-facing UI
```

### 6.2 Ingestion flow

1. **Discover PDFs**
   - Walk the categorized folders under `data/raw/`.
   - Join each file to the inventory metadata.

2. **Check for duplicates and changes**
   - Compute a SHA-256 hash.
   - If the document succeeded before and the hash is unchanged, skip it.
   - If the file changed, rebuild its chunks and replace its old vectors.

3. **Extract pages**
   - PyMuPDF returns text for each page.
   - The extractor preserves page numbers.

4. **OCR when needed**
   - If a page has too little usable text, or the inventory requires OCR, render the page to an image and run Tesseract.
   - `GRD-008` and `GRD-009` are forced through OCR because their embedded text layer is garbled.

5. **Clean**
   - Remove repeated headers/footers and page numbers conservatively.
   - Keep the actual scheme, crop, table, and procedure text.

6. **Chunk**
   - Create sliding 200-word windows with 100-word overlap.
   - Record `page_start` and `page_end`.

7. **Embed**
   - Prepend the document title and scheme/crop to the chunk text.
   - Encode the combined text into a 384-dimensional vector.
   - Store the original chunk text unchanged.

8. **Index**
   - Upsert chunk text, embeddings, and metadata into ChromaDB.
   - Delete any stale chunk IDs for that document.
   - Update `storage/ingestion_manifest.json`.

### 6.3 Question-answering flow

1. The user selects Marathi, Hindi, or English.
2. The question is length-checked and privacy-redacted where appropriate.
3. Known scheme/crop names are expanded for retrieval.
4. The question is embedded with the same multilingual model.
5. ChromaDB returns the highest-scoring chunks.
6. The optional threshold removes weak matches.
7. Per-document diversification prevents one document from filling the context.
8. The context builder assigns `[S1]`, `[S2]`, etc., and includes title, filename, page, category, and validity status.
9. Gemini or Ollama receives the grounded prompt.
10. The chain checks source markers.
11. The UI shows the answer and the supporting source cards.
12. If the answer is unsupported, the localized abstention message is returned.

### 6.4 Code map

```text
kisan-sahayak/
├── app.py                         # Streamlit UI
├── ingest.py                      # Ingestion and diagnostics CLI
├── requirements.txt
├── pytest.ini
├── .env.example
├── README.md
├── config/
│   ├── settings.py                # Pydantic settings, paths, validation
│   └── prompts.py                 # System prompt and localized messages
├── data/
│   ├── raw/                       # 26 indexed PDFs, categorized
│   └── metadata/
│       └── document_inventory.csv # 29 rows: 26 included + 3 duplicates
├── evaluation/
│   ├── questions.jsonl            # 30 questions: 10 per language
│   └── run_evaluation.py          # Retrieval and live evaluation
├── src/
│   ├── ingestion/
│   │   ├── loader.py              # Discovery + inventory
│   │   ├── extractor.py           # PyMuPDF page extraction
│   │   ├── ocr.py                 # Local Tesseract integration
│   │   ├── cleaner.py             # Conservative cleaning
│   │   ├── chunker.py             # Sliding word windows
│   │   └── pipeline.py            # Manifest + incremental indexing
│   ├── embeddings/
│   │   └── embedder.py            # Multilingual embedding wrapper
│   ├── vectorstore/
│   │   └── chroma_store.py        # ChromaDB operations
│   ├── retrieval/
│   │   ├── retriever.py           # Top-k, threshold, diversification
│   │   ├── query_expansion.py     # Marathi/Hindi scheme-name aliases
│   │   └── context_builder.py     # Bounded context with source IDs
│   ├── llm/
│   │   ├── base.py                # Provider interface
│   │   ├── gemini.py              # google-genai provider
│   │   ├── ollama.py              # Optional local provider
│   │   └── factory.py             # Explicit provider selection
│   ├── rag/
│   │   ├── chain.py               # ask_question(), citation validation
│   │   └── prompts.py             # RAG user-message builder
│   ├── services/
│   │   └── chat_service.py        # UI-facing service
│   └── utils/                     # Logging, text, redaction, language
├── storage/                       # Generated ChromaDB + manifest
├── logs/                          # Generated logs
└── tests/                         # 59 automated tests
```

---

## 7. Technology stack and why these choices were made

| Area | Choice | Why it was chosen | Trade-off |
|---|---|---|---|
| Language | Python 3.11 | Strong ecosystem for PDFs, embeddings, ML, and Streamlit; stable for the dependencies | Requires the project virtual environment |
| PDF extraction | PyMuPDF | Fast, reliable page-level extraction with good handling of government PDFs | Some scanned/legacy PDFs still need OCR |
| OCR | Tesseract 5.5 | Free, offline, supports English, Hindi, and Marathi | Slow on scanned pages; OCR errors are possible |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual, compact, CPU-friendly, works well enough for semantic search; can run without a paid embedding API | 128-token limit; handled with sliding windows |
| Vector database | ChromaDB | Persistent, local, simple metadata filtering, easy to inspect | Not designed for very large distributed workloads; 2,240 chunks is comfortably within its scope |
| Hosted LLM | Gemini via `google-genai` | Strong multilingual generation and a maintained official SDK | Requires an API key and quota; free-tier privacy constraints |
| Local LLM option | Ollama | Allows offline or privacy-sensitive experimentation | User must start Ollama and choose a suitable local model |
| UI | Streamlit | Rapid Python-first UI with chat widgets, sources, and sidebar controls | Not a full production front-end framework |
| Configuration | Pydantic Settings | Typed validation, `.env` support, clear failure messages | Configuration must be kept out of version control |
| Testing | pytest | Standard, readable, fixture-friendly | Real embedding tests take longer than fake-LLM tests |
| Retrieval | Direct Chroma cosine search + custom reranking | Transparent and easy to debug; no framework magic | Less feature-rich than a dedicated search framework |
| Orchestration | Custom Python pipeline | No LangChain; full control over prompts, citations, abstention, and data flow | More code to maintain directly |

### 7.1 Why this embedding model?

`paraphrase-multilingual-MiniLM-L12-v2` was chosen because:

- It supports English, Hindi, and Marathi in one shared vector space.
- It is small enough to run on CPU.
- It can be used fully offline after the first model download.
- It is a well-known Sentence Transformers model, which makes the embedding wrapper simple.
- It produces normalized vectors, making cosine similarity straightforward.

Its main limitation is the **128-token maximum sequence length**. That limitation directly shaped the chunking design.

This project therefore does not claim that the model embeds an entire 200-word chunk. Instead, the 100-word overlap ensures that later parts of a document are also represented near the beginning of another chunk.

### 7.2 Why ChromaDB?

ChromaDB was chosen because it is:

- Persistent and local
- Easy to run in a student project
- Able to store document text, vectors, and metadata together
- Compatible with metadata filters
- Easy to inspect and rebuild

For 2,240 chunks, a simpler local vector database is sufficient. A larger production system might use a managed vector database or a distributed search engine.

### 7.3 Why not use LangChain?

The project intentionally avoids LangChain. The data and answer flow are implemented directly so that:

- Retrieval is easy to understand.
- Citation rules are explicit.
- Abstention behavior is visible.
- Failure modes are easier to debug.
- There are fewer framework abstractions between the PDF and the answer.

### 7.4 Why not fine-tune a model?

Fine-tuning was not required for the task. The main problem was not model knowledge; it was **grounded retrieval and citation**. RAG addresses that directly. Fine-tuning would add training data requirements, evaluation complexity, and maintenance cost without solving the retrieval problem.

### 7.5 Why no agents or multi-agent workflows?

The task is a single grounded question-answering flow:

```text
retrieve → build context → answer → validate citations
```

An agent framework would add unnecessary complexity and more places for unsupported claims or uncontrolled tool use. The project uses a deterministic pipeline instead.

### 7.6 Why is there no hardcoded Gemini model ID?

Gemini model availability changes over time and depends on the API key. A hardcoded model can become unavailable or inaccessible. The project therefore:

- Requires `GEMINI_MODEL` in configuration.
- Provides `--list-models` to discover models available to the user’s key.
- Does not guess a model ID.
- Does not silently fall back to another provider.

### 7.7 Why keep source URLs blank?

The original download metadata containing official URLs was unavailable. Inventing a URL would be worse than showing no URL. The inventory keeps `source_url` blank unless a verified URL is added. The UI omits missing links.

---

## 8. The corpus and data preparation

### 8.1 Included documents

The corpus contains **26 unique PDFs** across five categories:

- Central Government schemes
- Maharashtra State schemes
- Crop advisories and integrated pest management
- Soil and fertilizer information
- Weather and agromet advisories

The metadata inventory has **29 rows**:

- 26 rows marked `include=true`
- 3 exact duplicate files marked `include=false`

The duplicate files remain untouched in the original download folder; they are not ingested.

### 8.2 Corpus statistics

| Statistic | Value |
|---|---:|
| Included documents | 26 |
| Unique pages | 851 |
| Extracted characters | approximately 2.30M |
| Indexed chunks after sliding windows | 2,240 |
| OCR pages | 97 |
| English documents | 22 |
| Marathi documents | 4 |
| Hindi documents | 0 |

### 8.3 OCR-heavy documents

Two documents require forced OCR because their text layer is garbled:

- `GRD-008` — PoCRA DBT Portal user manual
- `GRD-009` — MahaAgri-AI Policy 2025–2029

The local Tesseract installation is under:

```text
.local/tesseract/usr/bin/tesseract
.local/tesseract/usr/share/tesseract-ocr/5/tessdata/
```

Check it with:

```bash
.venv/bin/python ingest.py --ocr-health
```

Expected languages include `eng`, `hin`, and `mar`.

### 8.4 Data limitations

The project should be honest about what the corpus cannot prove:

- No Hindi source documents
- PM-KISAN documents describe 2018–19 rules
- Weather advisories are historical
- MahaAgri-AI is a state policy, not a farmer scheme
- Source URLs are blank
- Maha-DBT manual is screenshot-heavy
- Some screenshot content contains public demo beneficiary data

---

## 9. Ingestion and indexing

### 9.1 Stable chunk identity

Each chunk ID is derived from stable fields such as document ID, page information, chunk index, and text. This allows the pipeline to:

- Detect old chunks
- Add new chunks
- Delete stale chunks
- Avoid random duplicate IDs

### 9.2 The manifest

The manifest is stored at:

```text
storage/ingestion_manifest.json
```

For each document it records:

- Status
- SHA-256 hash
- Page count
- Chunk count
- OCR pages
- Warnings
- Errors
- Relevant metadata

This supports incremental ingestion. An unchanged successful document is skipped. A changed document is reprocessed and its old chunks are replaced.

### 9.3 Why title and scheme prefixes are embedded

A chunk might begin with a sentence like:

> “The benefit will be transferred in three instalments…”

Without context, the embedding may not know which scheme this belongs to. Prepending:

```text
Document: PM-KISAN FAQ
Scheme or crop: PM-KISAN
```

makes the embedding more useful. The stored text remains the original chunk so the LLM sees natural document language and the citation page remains correct.

### 9.4 Rebuilding the index

```bash
.venv/bin/python ingest.py --force
```

Use `--force` after changing chunking, embedding text, or document metadata that affects stored vectors.

Normal incremental ingestion:

```bash
.venv/bin/python ingest.py
```

Statistics only:

```bash
.venv/bin/python ingest.py --stats
```

---

## 10. Retrieval design

### 10.1 Query expansion

The corpus is mostly English, but users ask in Marathi and Hindi. The multilingual embedding model helps, but explicit scheme names and local abbreviations can still be missed.

The project therefore appends English retrieval names for known aliases. Examples:

| Input phrase | Added retrieval terms |
|---|---|
| पीएम किसान | PM-KISAN, Pradhan Mantri Kisan Samman Nidhi, farmer income support |
| पीक विमा / फसल बीमा | PMFBY, Pradhan Mantri Fasal Bima Yojana, crop insurance |
| कापूस / कपास | cotton, integrated pest management, IPM |
| महा-डीबीटी | Maha-DBT, Farmer Application Portal, registration, OTP |
| मृदा आरोग्य | Soil Health Card, soil sampling and testing |
| किसान क्रेडिट | Kisan Credit Card, KCC, farmer credit |

The aliases are retrieval aids only. They do not add answer facts and are not shown to the LLM as evidence. The list lives in `src/retrieval/query_expansion.py`.

For questions that ask for **methods, claim steps, or benefit amounts**, the retriever also sends a short English section-level hint as a second query. Examples include `cultural practices / mechanical practices / biocontrol practices` for cotton IPM, `claim intimation procedure documents` for PMFBY, and `interest subvention norms` for the Agriculture Infrastructure Fund. These hints name document sections or generic concepts; they do not contain amounts, deadlines, eligibility rules, or other answer facts.

### 10.2 Search and diversification

The retriever defaults to `top_k=5`.

It fetches extra candidates, then:

1. Applies the relevance threshold.
2. If an intent hint applies, keeps the best three results from the primary query and the best three from the hint query, merges them by chunk ID, and ranks the union.
3. Prefers at most `MAX_CHUNKS_PER_DOCUMENT=2` chunks from one document.
4. Fills the remaining slots from the best available chunks if diversification makes the list shorter than `top_k`.

This avoids two common RAG problems:

- One long document occupying the entire context window while other relevant documents are pushed out.
- A Marathi or Hindi question finding only the general definition of a topic, while the English document's actionable procedure section stays outside the top results.

This is intentionally a small, transparent query-fusion step, not an agent or a LangChain pipeline.

### 10.3 Why the threshold is 0.66

The retrieval evaluation measured:

- Lowest supported top score: **0.7200**
- Highest unsupported top score: **0.6590**

Threshold comparison on the current index:

| Threshold | Supported hit@5 | Supported recall | Unsupported questions returning chunks |
|---:|---:|---:|---:|
| 0.60 | 24/24 | 0.979 | 1/6 |
| 0.64 | 24/24 | 0.979 | 1/6 |
| **0.66** | **24/24** | **0.979** | **0/6** |
| 0.68 | 24/24 | 0.938 | 0/6 |
| 0.70 | 24/24 | 0.917 | 0/6 |

A threshold of **0.66** is therefore a reasonable starting point for this exact corpus and embedding model. It is not a universal constant. Recalibrate after changing the model or adding significantly different documents.

The LLM prompt still requires abstention even when chunks pass the threshold.

---

## 11. Answer generation, citations, and safety

### 11.1 LLM providers

The default provider is Gemini through the official `google-genai` SDK. Ollama is optional.

There is **no automatic fallback** between providers. If Gemini fails, the app shows an error instead of silently switching models.

### 11.2 Grounding prompt

The system prompt instructs the model to:

- Answer only from retrieved excerpts
- Never invent scheme names, benefits, amounts, deadlines, phone numbers, or application steps
- Answer in the user’s language
- Use simple language for farmers
- Use the exact not-found message when evidence is insufficient
- Distinguish central and state schemes
- Warn about historical or undated documents
- Cite claims with `[S1]`, `[S2]`, etc.
- Treat document text as untrusted reference data, not instructions
- Avoid asking for or repeating private identifiers
- Avoid personalized financial advice
- Explain when live weather, prices, deadlines, or eligibility cannot be verified

### 11.3 Citation validation

The app:

- Parses source markers in the model’s answer.
- Removes invalid source IDs.
- Shows only sources that were actually retrieved.
- Marks an answer as `unverified` if no valid citation is present.
- Returns no supporting sources for the localized abstention message.

### 11.4 Source URL behavior

Every current `source_url` is blank. The app therefore omits URLs. If verified URLs are added to the inventory and the document is re-ingested, the source cards can show them. No URL is generated by the model.

### 11.5 Privacy and redaction

Before sending text to the LLM, the pipeline applies best-effort redaction for patterns resembling:

- Aadhaar numbers
- Bank account numbers
- Phone numbers
- OTPs
- Email addresses

This is defense in depth, not a guarantee. Do not enter private farmer data into a free-tier hosted model.

### 11.6 Handling failures

The app distinguishes:

- Missing configuration
- Invalid question
- Retrieval failure
- Empty collection
- LLM quota/network/authentication failure

Each has a localized user-facing message rather than a raw stack trace.

---

## 12. Streamlit user interface

The UI includes:

- Language selector: मराठी, हिंदी, English
- Chat input
- Suggested questions
- Source cards with title, filename, page, category, validity, and optional URL
- Indexed chunk count and collection health
- Clear configuration errors if Gemini or Ollama is not set up

The app can start without a Gemini key. It will show a configuration message instead of generating an answer. That is intentional: configuration failures should be understandable, not hidden.

---

## 13. How to run the project

### 13.1 Requirements

- Python 3.11
- A local virtual environment at `.venv`
- Tesseract OCR with Marathi, Hindi, and English language data for ingestion
- Internet access for the first embedding-model download
- Internet access for hosted Gemini, or a running local Ollama server
- No GPU required

On Linux/macOS use `.venv/bin/python` and `.venv/bin/streamlit`. On Windows use `.venv\Scripts\python.exe` and `.venv\Scripts\streamlit.exe`.

### 13.2 Setup

For a fresh clone, follow the full [Quick start (after cloning)](#quick-start-after-cloning) above. In short:

```bash
cd kisan-sahayak

python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

cp .env.example .env
```

Never commit `.env`.

### 13.3 Configure Gemini

Add to `.env`:

```env
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=
```

Then discover models available to your key:

```bash
.venv/bin/python ingest.py --list-models
```

Set `GEMINI_MODEL` to one ID printed by that command. Do not guess a model name.

To use Ollama instead:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

Start Ollama separately with `ollama serve`.

### 13.4 Get the PDF corpus on a new machine

The PDFs are intentionally excluded from Git because they total about 71 MB. If you clone this project on another machine, obtain the team `data/raw` archive from the shared drive/college storage and extract it at the project root. See `data/raw/README.md` for the exact folder layout, restoration steps, and Git LFS alternative.

The generated ChromaDB index is also excluded. Each teammate should build it locally by running ingestion once.

### 13.5 Ingest documents

```bash
# Normal incremental ingestion
.venv/bin/python ingest.py

# Force a full rebuild after changing chunking or embedding text
.venv/bin/python ingest.py --force

# Process selected documents
.venv/bin/python ingest.py --document GRD-001 --document GRD-027

# Check collection count
.venv/bin/python ingest.py --stats

# Check OCR
.venv/bin/python ingest.py --ocr-health
```

Expected current collection count:

```text
2240
```

### 13.6 Run the Streamlit app

```bash
.venv/bin/streamlit run app.py
```

Open the local URL printed by Streamlit, normally:

```text
http://localhost:8501
```

### 13.7 No-API mode

Without `.env` or `GEMINI_API_KEY`:

- Ingestion works
- Collection statistics work
- Tests work
- Retrieval evaluation works
- The app starts
- The app refuses to generate answers and shows a configuration error

This is useful for demonstrating the retrieval and engineering work before live LLM access is available.

---

## 14. How to test the project

### 14.1 Full automated test suite

```bash
.venv/bin/python -m pytest -q
```

The current suite has **59 tests** and covers:

- PDF extraction and page metadata
- Blank and invalid PDFs
- OCR behavior
- Cleaning
- Chunking, exact 100-word overlap, page metadata
- Stable chunk IDs
- Real embedding dimensions and normalization
- ChromaDB operations
- Retrieval and thresholds
- Per-document diversification and fill behavior
- Marathi/Hindi query expansion, intent hints, and false-positive guards
- RAG citations, abstention, and failure handling
- Chat-service errors
- Incremental ingestion and stale-chunk replacement
- Evaluation dataset shape and language balance

Tests use fake LLMs and do not consume Gemini quota. The embedding tests load the real local model on CPU.

### 14.2 Dependency check

```bash
.venv/bin/pip check
```

Expected:

```text
No broken requirements found.
```

### 14.3 Retrieval evaluation

No API key is needed:

```bash
# Inspect raw scores without a threshold
.venv/bin/python evaluation/run_evaluation.py --top-k 5 --threshold -1.0

# Evaluate with the calibrated threshold
.venv/bin/python evaluation/run_evaluation.py --top-k 5 --threshold 0.66
```

The script writes a JSON result under `evaluation/results/` and prints:

- Overall hit rate
- MRR
- Expected-document recall
- Per-language metrics
- Per-question top score and rank

### 14.4 Live answer evaluation

Requires a configured Gemini key and model:

```bash
.venv/bin/python evaluation/run_evaluation.py --live --limit 5
```

Full evaluation consumes quota:

```bash
.venv/bin/python evaluation/run_evaluation.py --live
```

Live mode checks:

- Answer language
- Citation presence and validity
- Whether cited documents match expected documents
- Keyword coverage
- Abstention on unsupported questions

Start with `--limit 5` to control cost.

### 14.5 Useful manual checks

```bash
# Confirm the index survived a restart
.venv/bin/python ingest.py --stats

# Confirm OCR languages
.venv/bin/python ingest.py --ocr-health

# Confirm unchanged documents are skipped
.venv/bin/python ingest.py
```

---

## 15. How to demonstrate it to your guide

The strongest demonstration has two parts: **offline engineering evidence** and, if credentials are available, **live answer behavior**.

### 15.1 Five-minute demo without an API key

This is the safest route if you do not want to configure Gemini in front of your guide.

#### Step 1 — Show the project and architecture

Open `README.md` and point to:

- The architecture diagram
- The ingestion flow
- The question-answering flow
- The module map

Explain that the project does not train a model; it builds a custom document and retrieval pipeline around pretrained embeddings and a language model.

#### Step 2 — Show the corpus status

```bash
cd kisan-sahayak
.venv/bin/python ingest.py --stats
```

Point out:

- 26 documents
- 2,240 chunks
- collection name `farmer_knowledge`

If you want to show the manifest:

```bash
.venv/bin/python - <<'PY'
import json
m = json.load(open('storage/ingestion_manifest.json'))
print('documents:', len(m['documents']))
print('chunks:', sum(v.get('chunks', 0) for v in m['documents'].values()))
print('ocr pages:', sum(len(v.get('ocr_pages', [])) for v in m['documents'].values()))
PY
```

#### Step 3 — Run the tests

```bash
.venv/bin/python -m pytest -q
```

Say:

> “The automated suite covers extraction, OCR, chunking, embeddings, vector storage, retrieval, citation validation, abstention, privacy, and incremental ingestion.”

#### Step 4 — Run the retrieval evaluation

```bash
.venv/bin/python evaluation/run_evaluation.py --top-k 5 --threshold 0.66
```

Point to:

- 100% hit@5
- MRR 1.000
- 0.979 expected-document recall
- All six unsupported questions filtered

Be precise:

> “This is document-level retrieval evaluation. It proves the expected source is found. Live answer correctness still needs LLM evaluation.”

#### Step 5 — Show a source excerpt

You can demonstrate retrieval without the UI by using a small Python snippet:

```bash
.venv/bin/python - <<'PY'
from config.settings import get_settings
from src.embeddings.embedder import get_embedding_service
from src.retrieval.retriever import Retriever
from src.vectorstore.chroma_store import ChromaStore

settings = get_settings()
store = ChromaStore(settings)
embedder = get_embedding_service(
    settings.embedding_model,
    settings.embedding_batch_size,
    settings.embedding_device,
)
retriever = Retriever(settings, store, embedder)

question = "What is PM-KISAN and how much financial support does it provide?"
for rank, chunk in enumerate(retriever.retrieve(question, top_k=5), 1):
    meta = chunk.metadata
    print(rank, meta.get("document_id"), meta.get("page_number"), round(chunk.score, 3))
    print(chunk.text[:300].replace("\n", " "))
    print()
PY
```

This shows the metadata and page-level excerpts that the LLM would receive.

#### Step 6 — Show the app’s configuration behavior

```bash
.venv/bin/streamlit run app.py
```

Without a key, show that the app starts and gives a clear configuration message. This demonstrates deliberate failure handling.

### 15.2 Demo with live Gemini

If credentials are configured:

```bash
cp .env.example .env
# edit .env
.venv/bin/python ingest.py --list-models
# put the chosen model in GEMINI_MODEL
.venv/bin/streamlit run app.py
```

Ask these in order:

| Language | Question | What to show |
|---|---|---|
| Marathi | पीएम किसान योजनेचा उद्देश काय आहे आणि योजनेत किती आर्थिक सहाय्य मिळते? | Marathi answer, PM-KISAN sources, historical caveat |
| Hindi | पीएम किसान योजना के लिए कौन पात्र है? | Hindi answer and cross-lingual sources |
| English | What documents are required for PM-KISAN? | Required-document answer and page citations |
| English | How does integrated pest management help control pests in cotton? | Cotton IPM sources from `GRD-040` |
| English | How does a farmer register and log in to the Maha-DBT portal? | Maha-DBT manual sources |
| Unsupported | What is the current MSP for wheat in 2026? | Refusal instead of a fabricated current MSP |
| Unsupported | आज पुण्यात पाऊस पडेल का? | Refusal rather than a fake live weather forecast |

For each answer, expand **Supporting sources** and point out:

- title
- filename
- page number or page range
- validity status
- blank URL when no official URL exists

### 15.3 What to say about your contribution

A concise explanation:

> “I built the data ingestion and evaluation pipeline around pretrained tools. I curated the document corpus, wrote the page-aware extraction and OCR fallback flow, implemented sliding overlapping chunks with stable IDs and an incremental manifest, integrated a local multilingual embedding model with ChromaDB, added Marathi/Hindi scheme-name and section-intent query expansion, designed per-document retrieval diversification, built the grounded RAG chain with citation validation and abstention, and created the Streamlit UI and 30-question evaluation set. I did not train the embedding model or the LLM.”

### 15.4 What to show as evidence

- `storage/ingestion_manifest.json`
- Chroma count output
- `evaluation/results/retrieval_*.json`
- pytest output
- `storage/chroma_db/`
- page-level source cards in the UI
- the architecture diagram in this README

### 15.5 What not to claim

Do not claim:

- “The LLM is 100% accurate.”
- “The retrieval hit rate proves answer correctness.”
- “The weather information is current.”
- “The PM-KISAN rules are current.”
- “Source links are official” when the URL is blank.
- “I trained the embedding model.”
- “The system works for every Indian scheme.” It only handles the indexed corpus.

---

## 16. What you can do with it right now

### Works now without an API key

- Inspect the document corpus and inventory
- Rebuild the local vector index
- Check OCR health
- Run all 59 tests
- Run the retrieval evaluation
- Inspect retrieved chunks, page numbers, and scores
- Demonstrate the architecture and engineering decisions
- Add documents and test ingestion
- Start the Streamlit app and see its configuration behavior
- Use the local Ollama path if Ollama and a model are available

### Works after configuring Gemini

- Ask questions in Marathi, Hindi, or English
- Receive grounded answers with source citations
- Inspect supporting page-level sources
- Test abstention on unsupported questions
- Evaluate answer language and citation behavior
- Demonstrate the full RAG experience to your guide

### Not available as of now

- Live weather or market prices
- Current scheme deadlines unless present in the indexed documents
- Verified official links for the current corpus
- Hindi-source-native answers
- Production authentication or multi-user support
- Guaranteed answer correctness without live evaluation

---

## 17. Evaluation results

### 17.1 Retrieval evaluation

Dataset: 30 questions

- 10 Marathi
- 10 Hindi
- 10 English
- 24 answerable
- 6 unsupported

Results on the current index with no threshold filter:

| Metric | Result |
|---|---:|
| Hit@5 | **100% (24/24)** |
| MRR@5 | **1.000** |
| Expected-document recall@5 | **0.979** |
| Mean expected-document score | **0.7822** |
| Marathi hit rate | 100% |
| Hindi hit rate | 100% |
| English hit rate | 100% |

Before the retrieval fixes, the hit rate was approximately **58%**. The improvements were:

- Smaller sliding windows
- Title/scheme-aware embedding text
- Exact 100-word overlap
- Multilingual scheme-name expansion
- Section-level intent hints for procedures and scheme benefits
- Per-document diversification

The final evidence audit also confirmed that the top-5 context now includes:

- Cotton IPM cultural, mechanical, and biocontrol practices — `GRD-040`, page 36
- PMFBY claim-intimation and loss-assessment steps — `GRD-005`, pages 54–61
- Agriculture Infrastructure Fund 3% interest-subvention clause — `AIF-001`, pages 5–6

### 17.2 Threshold separation

| Group | Top score range |
|---|---:|
| Answerable questions | 0.7200 – 0.8526 |
| Unsupported questions | 0.3595 – 0.6590 |

A threshold of **0.66** cleanly separates the current evaluation set.

### 17.3 Answer-level evaluation

Not yet run in this workspace because no LLM credentials are configured. The live evaluator exists and can be run after setup:

```bash
.venv/bin/python evaluation/run_evaluation.py --live --limit 5
```

Until that is done, report retrieval results and deliberately avoid claiming verified answer accuracy.

---

## 18. What is still remaining

### Completed in local verification

- The project guide/README was expanded with architecture, concepts, technology choices, operation, testing, and a demonstration script.
- `.env.example` now contains the calibrated **0.66** threshold.
- The final 2,240-chunk index was smoke-tested with Streamlit AppTest:
  - no-key mode showed the expected configuration message;
  - dummy-key mode started with no exceptions and a chat input present.
- Incremental ingestion was confirmed:
  - 26 unchanged documents skipped;
  - 0 documents processed;
  - 0 failures;
  - collection count remained 2,240.

### Completed — Live Gemini verification

- The API key was moved into the Git-ignored `.env`; `.env.example` is blank again. The key was never printed in logs or answers.
- `ingest.py --list-models` loaded the available model list successfully.
- `GEMINI_MODEL=gemini-3.8-flash` produced correct, well-cited answers. The full run then hit the free tier's **20 requests/day** limit, plus a few temporary `504`/`503` responses. Those failures were returned as localized, graceful `llm_error` messages, not crashes.
- `GEMINI_MODEL=gemini-3.5-flash-lite` had available quota and was used to finish verification. Across both models, **all 24 answerable questions** produced at least one live `success` answer with verified citations (10 with `gemini-3.8-flash`, 14 with `gemini-3.5-flash-lite`). The workspace `.env` currently uses `gemini-3.5-flash-lite` so the demo works immediately; you may set any listed model explicitly.
- **Abstention:** all six unsupported questions returned the configured not-found response in the full run, with zero LLM calls for those questions.
- **Citations:** validation accepts both `[S1]` and combined markers such as `[S1, S2, S5]`. Invalid IDs are stripped; uncited answers get one explicit retry and are then safely rejected instead of being shown.
- **Content checks:** live answers included the PMFBY 72-hour intimation and claim steps, cotton IPM cultural/mechanical/biocontrol practices, soil sampling at 15–20 cm with a V-shaped cut, AIF 3% interest subvention / ₹2 crore cap / 7-year period, Maha-DBT registration and OTP steps, and MahaAgri-AI 2025–2029 objectives.
- **Sources:** source cards showed real document titles and page numbers. The only URL observed in an answer (`mahadbt.maharashtra.gov.in/Farmer`) appears verbatim in `GRD-027`, page 2; no links were fabricated.
- **Historical caveats:** PM-KISAN answers stated that the available documents describe 2018–19 rules and told users to verify current rules.
- **Stability:** Marathi, Hindi, and English script checks passed for all live answers that were generated.

### Remaining optional improvements

- Add verified official URLs to the inventory
- Add Hindi source documents
- Add current scheme guidelines
- Add more evaluation questions
- Add a longer paid-tier live run with a single model
- Add answer-level expert review
- Initialize Git and commit the project
- Add screenshots to the guide
- Add a small demo video or slide deck

---

## 19. Known limitations

1. **No Hindi source documents.** The corpus has 22 English and 4 Marathi documents. Hindi answers are cross-lingual.
2. **PM-KISAN documents describe 2018–19 rules.** Present them as historical and tell users to verify current rules.
3. **Weather advisories are historical.** They cover January and March 2026 and are not live forecasts.
4. **MahaAgri-AI Policy is not a farmer scheme.** It is included as a state policy document but may add noise.
5. **No official source URLs are recorded.** The UI omits links rather than inventing them.
6. **Three duplicates are excluded.** They remain in the original download folder and are marked `include=false`.
7. **Maha-DBT manual is screenshot-heavy.** OCR may miss screenshot-only details.
8. **Public demo beneficiary data appears in screenshots.** Context is redacted best-effort, but the original PDF is unchanged.
9. **The embedding model has a 128-token limit.** Sliding windows mitigate this but do not make the model see entire documents at once.
10. **Retrieval hit rate is not answer accuracy.** Live answer generation was verified across all 24 answerable questions, but that is not the same as an expert fact-check of every generated claim.

---

## 20. Security and privacy

- API keys are read from `.env` or environment variables.
- `.env`, `.venv`, `.local/`, `storage/`, logs, and local PDFs are excluded by `.gitignore`.
- Keys are not written to logs or shown in the UI.
- The app does not ask for Aadhaar numbers, bank details, OTPs, or passwords.
- Retrieved document text is labeled as untrusted reference data to reduce prompt-injection risk.
- Redaction is best-effort, not a guarantee.
- Do not send private farmer data to a free-tier hosted model.
- A hosted LLM provider may process retrieved excerpts; review its privacy terms before using real farmer data.
- The Ollama path can be used for local experimentation when appropriate.

---

## 21. Troubleshooting

### `GEMINI_API_KEY is not set`

Create `.env` from `.env.example`, add the key, and restart Streamlit.

### `GEMINI_MODEL is not set`

Run:

```bash
.venv/bin/python ingest.py --list-models
```

Choose a model from the returned list. Do not guess.

### The collection is empty

```bash
.venv/bin/python ingest.py
.venv/bin/python ingest.py --stats
```

### OCR is unavailable

```bash
.venv/bin/python ingest.py --ocr-health
```

Check that `.local/tesseract` exists or install system Tesseract with `eng`, `hin`, and `mar`.

### The first embedding run is slow

The model downloads on first use. Later runs use the local cache.

### `RELEVANCE_THRESHOLD` causes refusals

The application defaults to the measured value **0.66** for the current index. Set it to a blank value only while inspecting raw scores. Lower it if supported questions are being rejected. Recalibrate after changing the model or corpus.

### Streamlit port is busy

```bash
.venv/bin/streamlit run app.py --server.port 8502
```

### Ollama is not running

```bash
ollama serve
```

Then confirm `LLM_PROVIDER=ollama` and that the selected model is installed.

---

## 22. Commands at a glance

```bash
# After cloning and restoring data/raw (see Quick start above)
cd kisan-sahayak

# Environment and health
.venv/bin/python --version
.venv/bin/pip check
.venv/bin/python ingest.py --stats
.venv/bin/python ingest.py --ocr-health

# Ingestion
.venv/bin/python ingest.py
.venv/bin/python ingest.py --force
.venv/bin/python ingest.py --document GRD-001

# Tests
.venv/bin/python -m pytest -q

# Retrieval evaluation
.venv/bin/python evaluation/run_evaluation.py --top-k 5 --threshold -1.0
.venv/bin/python evaluation/run_evaluation.py --top-k 5 --threshold 0.66

# Live evaluation (requires Gemini configuration)
.venv/bin/python ingest.py --list-models
.venv/bin/python evaluation/run_evaluation.py --live --limit 5

# Run the app
.venv/bin/streamlit run app.py
```

---

**Bottom line:** Kisan Sahayak is a working local RAG system with a real document index, a tested ingestion and retrieval pipeline, multilingual query handling, source citations, abstention behavior, and a Streamlit demonstration. Retrieval and live Gemini answer generation have both been verified end to end. New teammates should start with the [Quick start (after cloning)](#quick-start-after-cloning) section above.
