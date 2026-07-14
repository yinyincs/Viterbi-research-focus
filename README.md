# Google Scholar Research Landscape Analysis

Automatically build a department-level research landscape from Google Scholar.

This repository provides an end-to-end pipeline to:

1. Scrape recent publications from Google Scholar
2. Infer representative research keywords for each professor using a local LLM (Ollama)
3. Cluster professors based on their research interests and generate interactive visualizations of the department research landscape

---

## Pipeline

```
Google Scholar
        │
        ▼
google_scholar_publication.py
        │
        ▼
Professor publication CSV files
        │
        ▼
infer_keywords_local_ollama_folder_domain_top20_unranked.py
        │
        ▼
Department keyword summary
(Book1_google_scholar.csv)
        │
        ▼
viterbi_hdbscan_centroid_label.py
        │
        ▼
Research landscape
Clusters
Interactive visualizations
Statistics
```

---

# Step 1 — Scrape Google Scholar Publications

Script:

```
google_scholar_publication.py
```

### Purpose

Scrape publication titles from a single Google Scholar profile.

The script

- opens one Google Scholar profile
- extracts the scholar name
- collects publications within a specified year range
- saves a CSV file automatically named after the scholar
- closes the browser when scraping is complete

Default year range:

```
2024–2026
```

## Installation

```bash
pip install pandas playwright
playwright install chromium
```

## Configure

Edit:

```python
SCHOLAR_URL
```

and

```python
output_dir
```

Optionally modify

```python
START_YEAR
END_YEAR
```

## Run

```bash
python google_scholar_publication.py
```

Output:

```
publication search and keyword extraction/
    Biomedical Engineering/
        John_Smith_publications_2024_2026.csv
        Jane_Doe_publications_2024_2026.csv
        ...
```

The browser profile is persisted between runs to reduce the likelihood of Google Scholar CAPTCHAs.

---

# Step 2 — Infer Research Keywords

Script:

```
infer_keywords_local_ollama_folder_domain_top20_unranked.py
```

### Purpose

Infer representative research keywords for each professor using a local Large Language Model.

The script

- reads every publication CSV in a department folder
- summarizes all publication titles together
- generates **20 representative research keywords**
- automatically infers the research domain from the folder name
- exports keyword summaries for every professor
- generates a department-wide keyword summary

Example:

```
Publication titles
        ↓
Local LLM (Ollama)
        ↓
20 representative research keywords
```

## Requirements

Install Ollama

```bash
brew install ollama
```

Start the server

```bash
ollama serve
```

Download a model

```bash
ollama pull qwen3:8b
```

Install Python packages

```bash
pip install pandas openpyxl
```

## Configure

Edit

```python
INPUT_FOLDER
```

Example

```python
INPUT_FOLDER = Path(
    "publication search and keyword extraction/Biomedical Engineering"
)
```

The domain is automatically inferred from the folder name.

## Run

```bash
python infer_keywords_local_ollama_folder_domain_top20_unranked.py
```

Outputs

```
keyword_reviews/

    Professor_A_research_keywords.xlsx

    Professor_B_research_keywords.xlsx

    ...

    all_professor_keyword_summary.xlsx
```

The summary file contains one row per professor with

- scholar name
- research domain
- publication count
- 20 representative research keywords

---

# Step 3 — Department Research Landscape Clustering

Script

```
viterbi_hdbscan_centroid_label.py
```

### Purpose

Cluster professors based on their research interests and visualize the research landscape of an academic department.

The pipeline uses

```
Research keywords
        ↓
SentenceTransformer embeddings
        ↓
HDBSCAN clustering
        ↓
UMAP visualization
```

**Important**

HDBSCAN performs the clustering directly on the SentenceTransformer embeddings.

UMAP is used **only for visualization** and does **not** affect the clustering results.

---

## Features

- SentenceTransformer embeddings from research keywords
- HDBSCAN clustering
- Optional outlier reassignment
- Optional noise preservation
- Automatic cluster naming
- Interactive Plotly visualizations
- Static publication-quality figures

Cluster labels are generated using

- centroid-based representative keywords
- keyword-level c-TF-IDF

---

## Input

Example:

```
Book1_google_scholar.csv
```

Required columns

| Column | Description |
|---------|-------------|
| Department | Department name |
| PI/Lab / PI / Professor | Professor name |
| Research_Focus | Semicolon-separated research keywords |
| PhD_Students | Comma-separated student names (optional) |

---

## Configure

Edit the CONFIG section

```python
INPUT_PATH
OUTDIR
DEPARTMENT_NAME
```

---

## Run

```bash
python viterbi_hdbscan_centroid_label.py
```

---

# Outputs

The script generates approximately 15 visualizations, including

- UMAP research landscape
- Cluster distribution
- Cluster keyword summaries
- Cluster heatmaps
- PI-to-cluster tables
- PhD student distributions
- Cluster statistics
- Interactive Plotly dashboards
- Publication-quality PNG figures

---

# Overall Workflow

```
1.
Google Scholar
        │
        ▼
google_scholar_publication.py

↓

Publication CSVs

────────────────────────────

2.
infer_keywords_local_ollama_folder_domain_top20_unranked.py

↓

20 research keywords per professor

↓

Department keyword summary

────────────────────────────

3.
viterbi_hdbscan_centroid_label.py

↓

SentenceTransformer embeddings

↓

HDBSCAN clustering

↓

UMAP visualization

↓

Department research landscape
```

---

# Dependencies

- Python 3.10+
- Playwright
- Pandas
- OpenPyXL
- Ollama
- SentenceTransformers
- HDBSCAN
- UMAP
- Plotly
- Matplotlib
- Scikit-learn
- Kaleido (optional, for exporting Plotly figures)

---

# Citation

If you use this repository in academic work, please cite the repository and any associated publication describing the methodology.
