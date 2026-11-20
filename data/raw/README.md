# Source PDFs

For the complete first-run path, see `README.md` → **Quick start (after cloning)**.

The 26 source PDFs are intentionally **not committed to Git**. Their combined size is about 71 MB, and binary PDFs make repository clones and history unnecessarily heavy.

## What is tracked

The code, tests, evaluation questions, document inventory, and this README are tracked in Git:

```text
data/metadata/document_inventory.csv
README.md
```

The inventory contains the expected relative paths, hashes, titles, and categories for all 26 included documents.

## What is not tracked

```text
data/raw/**/*.pdf
storage/
logs/
.local/
.venv/
.env
```

The generated ChromaDB index, local Tesseract installation, virtual environment, logs, and API keys are also excluded.

## How group mates should restore the corpus

1. Get the team’s `data/raw` archive from the shared drive, college storage, or another agreed team location.
2. Extract it at the project root so the directory structure becomes:

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

3. Do not rename the PDF files. The inventory paths and SHA-256 hashes expect the original filenames.
4. Confirm the corpus and index:

```bash
cd kisan-sahayak
.venv/bin/python ingest.py --stats
.venv/bin/python ingest.py
```

The first run builds the local ChromaDB index. The second run should report all unchanged documents as skipped.

## If the team wants a one-command clone

PDFs can be stored with **Git LFS**, but that is optional. If the team chooses LFS:

```bash
git lfs install
git lfs track "data/raw/**/*.pdf"
git add .gitattributes
```

Then coordinate with the team because every collaborator must have Git LFS installed, and GitHub LFS storage/quota applies.

For this project, the recommended approach is:

- **GitHub private repository** for code and documentation
- **Shared drive/college storage** for the 71 MB `data/raw` archive
- Each teammate runs ingestion locally with their own environment

Never upload `.env`, API keys, `storage/`, `.venv/`, or `.local/` to GitHub.

## API keys

Each group member should create their own local `.env` from `.env.example`:

```bash
cp .env.example .env
```

Do not share a single Gemini API key in the repository or group chat. Use individual keys, or use the optional local Ollama provider when appropriate.
