#!/usr/bin/env python3
"""
Infer representative scholar-level research keywords from Google Scholar publication-title CSV files.

Input:
  - A folder of professor publication CSV files, e.g.:
      publication search and keyword extraction/Biomedical Engineering/*.csv
  - Each CSV should contain a publication title column, usually: title
  - In addition: scholar_name, scholar_id, year, google_scholar_link

Output:
  - For each professor:
      <INPUT_FOLDER>/keyword_reviews/<Scholar_Name>_research_keywords.xlsx

  - Each Excel file contains:
      1. Publications
         original CSV rows only
      2. Research_Keywords
         20 representative scholar-level research keywords, unranked

  - Also writes below summary for all the professors in the department
      <INPUT_FOLDER>/keyword_reviews/all_professor_keyword_summary.xlsx

How to run on Mac:
  1. Install Ollama:
       brew install ollama

  2. Start the Ollama server in one terminal:
       ollama serve

  3. Pull a local model:
       ollama pull qwen3:8b
       # or: ollama pull llama3.1:8b

  4. Install Python packages:
       pip install pandas openpyxl

  5. Open a separate terminal, make sure the Ollama server is still running, then run:
       python infer_keywords_local_ollama_folder_domain_top20_unranked.py

Notes:
  - This script uses Ollama's local HTTP API through Python standard library urllib.
  - Edit INPUT_FOLDER and OLLAMA_MODEL below if needed.
  - The research domain is automatically inferred from INPUT_FOLDER.name.
    Example: Biomedical Engineering, Computer Science, Mechanical Engineering, etc.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# =====================================================
# SETTINGS: EDIT THESE ONLY
# =====================================================

INPUT_FOLDER = Path("publication search and keyword extraction/AME1")
DOMAIN_NAME = INPUT_FOLDER.name

OUTPUT_SUBFOLDER_NAME = "keyword_reviews"

# good choice on Mac if you have enough RAM
OLLAMA_MODEL = "qwen3:8b"      
OLLAMA_URL = "http://localhost:11434/api/chat"

SUMMARY_KEYWORDS = 20
# small pause between local model calls
REQUEST_SLEEP_SECONDS = 0.2     
RETRY_COUNT = 2
OLLAMA_TIMEOUT_SECONDS = 240

# If True, process only first N files for testing. Use None to process all.
MAX_FILES: Optional[int] = None

# =====================================================
# BASIC UTILITIES
# =====================================================


def clean_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def safe_filename(name: object) -> str:
    name = clean_text(name) or "unknown_scholar"
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return safe or "unknown_scholar"


def find_title_column(df: pd.DataFrame) -> str:
    candidates = [
        "title",
        "publication_title",
        "Publication Title",
        "Title",
        "paper_title",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        normalized = col.lower().replace(" ", "_").strip()
        if normalized in {"title", "publication_title", "paper_title"}:
            return col
    raise ValueError(f"Cannot find title column. Available columns: {list(df.columns)}")


def find_scholar_name(df: pd.DataFrame, csv_path: Path) -> str:
    candidates = [
        "scholar_name",
        "name",
        "professor",
        "PI/Lab",
        "pi_lab",
    ]
    for col in candidates:
        if col in df.columns:
            values = df[col].dropna().astype(str).str.strip()
            values = values[values != ""]
            if not values.empty:
                return values.iloc[0]

    # Fallback: infer from filename like Assad_Oberai_publications_2024_2026.csv
    stem = csv_path.stem
    stem = re.sub(r"_publications?_?\d{4}_?\d{4}$", "", stem, flags=re.I)
    return stem.replace("_", " ").strip() or csv_path.stem


def try_parse_json(text: str) -> Any:
    """Parse JSON even if the model wraps it in Markdown fences or extra text."""
    text = text.strip()

    # Remove Markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Direct parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first JSON object/list.
    starts = [i for i in [text.find("{"), text.find("[")] if i != -1]
    if not starts:
        raise ValueError(f"No JSON found in model response: {text[:300]}")

    start = min(starts)
    candidate = text[start:]

    # Try object/list ending.
    for end_char in ["}", "]"]:
        end = candidate.rfind(end_char)
        if end != -1:
            snippet = candidate[: end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from model response: {text[:300]}")


def normalize_keyword_list(items: Any, max_keywords: int) -> List[str]:
    """Clean and deduplicate a model-produced keyword list."""
    if isinstance(items, dict):
        for key in [
            "research_keywords",
            "keywords",
            "summary_keywords",
            "primary_research_keywords",
            "top_20_keywords",
        ]:
            if key in items:
                items = items[key]
                break

    if not isinstance(items, list):
        return []

    generic_terms = {
        "learning",
        "model",
        "models",
        "method",
        "methods",
        "framework",
        "frameworks",
        "study",
        "studies",
        "analysis",
        "system",
        "systems",
        "approach",
        "approaches",
    }

    keywords: List[str] = []
    seen = set()

    for item in items:
        kw = clean_text(item).lower()
        kw = kw.strip(" .;,-")
        kw = re.sub(r"\s+", " ", kw)

        # Avoid very long sentence-like output.
        if not kw or len(kw.split()) > 6:
            continue

        # Avoid pure generic words.
        if kw in generic_terms:
            continue

        if kw not in seen:
            seen.add(kw)
            keywords.append(kw)

        if len(keywords) >= max_keywords:
            break

    return keywords

# =====================================================
# OLLAMA LOCAL LLM CALL
# =====================================================


def ollama_chat(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error: Optional[Exception] = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
                raw = response.read().decode("utf-8")
                obj = json.loads(raw)
                return obj["message"]["content"]
        except Exception as e:
            last_error = e
            print(f"    Ollama warning on attempt {attempt + 1}/{RETRY_COUNT + 1}: {repr(e)}")
            if attempt < RETRY_COUNT:
                time.sleep(2.0)
            else:
                break

    raise RuntimeError(
        "Ollama request failed. Make sure Ollama is running with `ollama serve` "
        f"and the model is installed with `ollama pull {OLLAMA_MODEL}`. "
        f"Last error: {last_error}"
    )

# =====================================================
# LLM PROMPTS
# Summarize a scholar's research areas using publication titles.
# Incorporate domain knowledge where appropriate.
# =====================================================


SUMMARY_SYSTEM_PROMPT = f"""
You are a senior researcher in the field of {DOMAIN_NAME}.

Your task is to identify representative research keywords that describe a scholar's
overall research agenda based on all publication titles.

Requirements:
1. Analyze all publication titles together.
2. Identify recurring technical research topics.
3. Merge synonymous concepts into a single keyword.
4. Prefer research themes, methodologies, and application areas.
5. Avoid generic words such as: model, method, framework, study, analysis, system.
6. Do not invent topics unsupported by the publication titles.
7. Use your expertise in {DOMAIN_NAME} to infer concise research topics.
8. Return exactly {SUMMARY_KEYWORDS} research keywords.
9. Do not rank the keywords.
10. Do not assign scores or frequencies.
11. Do not categorize the keywords.
12. Return JSON only.

Avoid:
- paper-specific phrases
- generic scientific terms
- publication titles copied verbatim

Format:
{{
  "research_keywords": [
    "keyword1",
    "keyword2",
    "keyword3"
  ]
}}
""".strip()


def build_summary_prompt(
    scholar_name: str,
    titles: List[str],
    domain_name: str,
) -> str:
    titles_text = "\n".join(
        f"{i + 1}. {title}"
        for i, title in enumerate(titles)
    )

    return f"""
Research Domain: {domain_name}

Scholar: {scholar_name}

Publication Titles:

{titles_text}

You are a senior researcher in {domain_name}.

Based on all publication titles, identify exactly {SUMMARY_KEYWORDS}
representative research keywords that collectively describe the scholar's
research agenda.

The keywords should be:
- concise
- technically meaningful
- representative of the scholar's overall research focus
- supported by the publication titles

Do not rank the keywords.
Do not assign scores.
Do not assign frequencies.
Do not categorize the keywords.
Return JSON only in this exact format:

{{
  "research_keywords": [
    "keyword1",
    "keyword2",
    "keyword3"
  ]
}}
""".strip()

# =====================================================
# PROCESSING
# =====================================================


def infer_summary_keywords(
    scholar_name: str,
    titles: List[str],
    domain_name: str,
) -> List[str]:
    prompt = build_summary_prompt(scholar_name, titles, domain_name)
    response = ollama_chat(SUMMARY_SYSTEM_PROMPT, prompt)
    parsed = try_parse_json(response)
    return normalize_keyword_list(parsed, SUMMARY_KEYWORDS)


def process_one_csv(csv_path: Path, output_dir: Path, domain_name: str) -> Optional[Dict[str, str]]:
    print(f"\nProcessing: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Skipped: could not read CSV: {e}")
        return None

    try:
        title_col = find_title_column(df)
    except ValueError as e:
        print(f"  Skipped: {e}")
        return None

    scholar_name = find_scholar_name(df, csv_path)
    safe_name = safe_filename(scholar_name)

    titles = []
    seen_titles = set()
    for title_value in df[title_col]:
        title = clean_text(title_value)
        title_key = title.lower()
        if title and title_key not in seen_titles:
            titles.append(title)
            seen_titles.add(title_key)

    if not titles:
        print("  Skipped: no valid publication titles found.")
        return None

    print(f"  Scholar: {scholar_name}")
    print(f"  Research domain from folder: {domain_name}")
    print(f"  Publication titles used: {len(titles)}")
    print("  Inferring 20 unranked representative research keywords...")

    try:
        research_keywords = infer_summary_keywords(scholar_name, titles, domain_name)
    except Exception as e:
        print(f"  Warning: keyword inference failed: {e}")
        research_keywords = []

    keywords_df = pd.DataFrame({
        "keyword": research_keywords,
    })

    metadata_df = pd.DataFrame({
        "scholar_name": [scholar_name],
        "research_domain": [domain_name],
        "source_file": [csv_path.name],
        "publication_titles_used": [len(titles)],
        "research_keywords": ["; ".join(research_keywords)],
    })

    output_file = output_dir / f"{safe_name}_research_keywords.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Publications", index=False)
        keywords_df.to_excel(writer, sheet_name="Research_Keywords", index=False, startrow=0)
        metadata_df.to_excel(
            writer,
            sheet_name="Research_Keywords",
            index=False,
            startrow=len(keywords_df) + 3,
        )

    print(f"  Saved: {output_file}")
    print(f"  Keywords: {'; '.join(research_keywords)}")

    time.sleep(REQUEST_SLEEP_SECONDS)

    return {
        "scholar_name": scholar_name,
        "research_domain": domain_name,
        "source_file": csv_path.name,
        "output_file": str(output_file),
        "publication_titles_used": str(len(titles)),
        "research_keywords": "; ".join(research_keywords),
    }


def main() -> None:
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER.resolve()}")

    output_dir = INPUT_FOLDER / OUTPUT_SUBFOLDER_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(INPUT_FOLDER.glob("*.csv"))
    if MAX_FILES is not None:
        csv_files = csv_files[:MAX_FILES]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {INPUT_FOLDER.resolve()}")

    print(f"Input folder: {INPUT_FOLDER.resolve()}")
    print(f"Research domain: {DOMAIN_NAME}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Ollama model: {OLLAMA_MODEL}")
    print(f"CSV files: {len(csv_files)}")

    all_rows: List[Dict[str, str]] = []

    for csv_path in csv_files:
        result = process_one_csv(csv_path, output_dir, DOMAIN_NAME)
        if result is not None:
            all_rows.append(result)

    if all_rows:
        combined_df = pd.DataFrame(all_rows)
        combined_output = output_dir / "all_professor_keyword_summary.xlsx"
        with pd.ExcelWriter(combined_output, engine="openpyxl") as writer:
            combined_df.to_excel(writer, sheet_name="Summary", index=False)
        print(f"\nCombined summary saved: {combined_output}")

    print(f"\nDone. Processed {len(all_rows)} / {len(csv_files)} CSV files.")


if __name__ == "__main__":
    main()
