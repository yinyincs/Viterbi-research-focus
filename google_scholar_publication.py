#!/usr/bin/env python3
"""
Scrape publication titles from ONE Google Scholar profile link each time for years 2024-2026.

What this script does:
  1. Opens one Google Scholar profile URL.
  2. Reads the scholar/profile name from the page.
  3. Scrapes publication titles from 2024, 2025, and 2026.
  4. Saves a CSV file whose filename includes the scholar name.
  5. Once the scraping is complete, the browser will close automatically.

Install once:
  pip install pandas playwright
  playwright install chromium

Run:
  python google_scholar_publication.py

"""

from __future__ import annotations

import csv
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs,urlparse

import pandas as pd
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


# ======================
# Edit this Scholar link
# ======================
SCHOLAR_URL = "https://scholar.google.com/citations?hl=en&user=onXXy6MAAAAJ"

# ======================
# Change the path if needed. 
# Make sure to have a unique folder name for each department. Change the name of the folder below accordingly.
# ======================
output_dir = Path("publication search and keyword extraction") / "AME1"
output_dir.mkdir(parents=True, exist_ok=True)

# ======================
# Edit the year range if needed
# ======================
START_YEAR = 2024
END_YEAR = 2026

# ======================
# Save the browser profile so Playwright reuses the same profile across runs.
# This helps reduce the likelihood of triggering Google Scholar CAPTCHAs.
# ======================
PROFILE_DIR = "./chrome_scholar_profile"
HEADLESS = False
MAX_SHOW_MORE_CLICKS = 10
DELAY_MIN = 2.0
DELAY_MAX = 5.0

DEFAULT_OUTPUT_CSV = "scholar_publications_2024_2026.csv"
PROFILE_URL_RE = re.compile(r"user=([^&?#]+)")


def clean_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def random_sleep(min_seconds: float = DELAY_MIN, max_seconds: float = DELAY_MAX) -> None:
    time.sleep(random.uniform(min_seconds, max_seconds))


def extract_scholar_id(link: str) -> str:
    """Extract Google Scholar user/profile ID from a Scholar profile URL."""
    link = clean_text(link)
    if not link:
        return ""

    parsed = urlparse(link)
    query = parse_qs(parsed.query)
    if "user" in query and query["user"]:
        return query["user"][0].strip()

    match = PROFILE_URL_RE.search(link)
    if match:
        return match.group(1).strip()

    return ""


def normalize_scholar_url(link: str, scholar_id: str) -> str:
    """Use Scholar's publication-date sorting page."""
    if scholar_id:
        return f"https://scholar.google.com/citations?user={scholar_id}&hl=en&sortby=pubdate&pagesize=100"
    return clean_text(link)


def parse_year(raw: str) -> Optional[int]:
    raw = clean_text(raw)
    match = re.search(r"(19|20)\d{2}", raw)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def safe_filename_name(name: str) -> str:
    """Convert scholar name to a safe filename component."""
    name = clean_text(name)
    if not name:
        return "unknown_scholar"
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return safe or "unknown_scholar"


def is_blocked_or_captcha(page) -> bool:
    """Detect common Google Scholar CAPTCHA/block pages."""
    try:
        text = page.locator("body").inner_text(timeout=3000).lower()
    except Exception:
        return False

    block_terms = [
        "our systems have detected unusual traffic",
        "not a robot",
        "captcha",
        "unusual traffic",
    ]
    return any(term in text for term in block_terms)


def get_scholar_name(page) -> str:
    """Extract the displayed Google Scholar profile name."""
    selectors = [
        "#gsc_prf_in",
        "div#gsc_prf_in",
    ]

    for selector in selectors:
        try:
            if page.locator(selector).count() > 0:
                name = clean_text(page.locator(selector).first.inner_text(timeout=5000))
                if name:
                    return name
        except Exception:
            continue

    return "Unknown Scholar"


def get_loaded_years(page) -> List[int]:
    """Return all years currently loaded in the Scholar publication table."""
    years: List[int] = []
    year_cells = page.locator("span.gsc_a_h")

    for i in range(year_cells.count()):
        try:
            year = parse_year(year_cells.nth(i).inner_text(timeout=1000))
            if year is not None:
                years.append(year)
        except Exception:
            continue

    return years


def should_stop_loading_more(page) -> bool:
    """
    Stop once at least one loaded row is older than START_YEAR.
    With sortby=pubdate, later loaded rows should be older.
    """
    years = get_loaded_years(page)
    if not years:
        return False
    return min(years) < START_YEAR


def click_show_more_until_enough(page) -> None:
    """Click Show More until rows older than START_YEAR appear, or button stops working."""
    for click_idx in range(MAX_SHOW_MORE_CLICKS):
        if should_stop_loading_more(page):
            return

        button = page.locator("#gsc_bpf_more")
        if button.count() == 0:
            return

        try:
            disabled = button.get_attribute("disabled")
            aria_disabled = button.get_attribute("aria-disabled")
            if disabled is not None or aria_disabled == "true":
                return
        except Exception:
            pass

        try:
            before_count = page.locator("tr.gsc_a_tr").count()
            print(f"Clicking Show more, attempt {click_idx + 1}")
            button.click(timeout=5000)
            page.wait_for_timeout(int(random.uniform(DELAY_MIN, DELAY_MAX) * 1000))
            after_count = page.locator("tr.gsc_a_tr").count()

            if after_count <= before_count:
                return
        except Exception as e:
            print(f"Could not click Show more: {e}")
            return


def scrape_loaded_rows(page, scholar_name: str, scholar_id: str, scholar_link: str) -> List[Dict[str, object]]:
    """Scrape currently loaded Scholar rows and keep 2024-2026 titles."""
    records: List[Dict[str, object]] = []
    rows = page.locator("tr.gsc_a_tr")

    for i in range(rows.count()):
        row = rows.nth(i)

        try:
            title = clean_text(row.locator("a.gsc_a_at").inner_text(timeout=2000))
        except Exception:
            title = ""

        try:
            year_text = clean_text(row.locator("span.gsc_a_h").inner_text(timeout=2000))
        except Exception:
            year_text = ""

        year = parse_year(year_text)

        if title and year is not None and START_YEAR <= year <= END_YEAR:
            records.append({
                "scholar_name": scholar_name,
                "scholar_id": scholar_id,
                "google_scholar_link": scholar_link,
                "year": year,
                "title": title,
            })

    return records


def scrape_one_profile(page, scholar_link: str) -> tuple[str, str, List[Dict[str, object]]]:
    """Open one Google Scholar profile and scrape publication titles."""
    scholar_id = extract_scholar_id(scholar_link)
    url = normalize_scholar_url(scholar_link, scholar_id)

    if not url:
        raise ValueError("Missing Scholar URL. Please edit SCHOLAR_URL near the top of the script.")

    print(f"Opening: {url}")

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(int(random.uniform(DELAY_MIN, DELAY_MAX) * 1000))
    except PlaywrightTimeoutError:
        raise RuntimeError("Timeout opening Google Scholar profile.")
    except Exception as e:
        raise RuntimeError(f"Failed opening Google Scholar profile: {e}")

    if is_blocked_or_captcha(page):
        print("Google Scholar may be showing CAPTCHA/block page.")
        print("Solve it in the browser window, then return to Terminal.")
        input("Press Enter after solving CAPTCHA, or Ctrl+C to stop: ")
        page.wait_for_timeout(3000)

    scholar_name = get_scholar_name(page)
    print(f"Scholar name: {scholar_name}")
    print(f"Scholar ID: {scholar_id}")

    try:
        page.wait_for_selector("tr.gsc_a_tr", timeout=15000)
    except PlaywrightTimeoutError:
        print("No publication rows found.")
        return scholar_name, scholar_id, []

    click_show_more_until_enough(page)

    seen = set()
    records: List[Dict[str, object]] = []

    for rec in scrape_loaded_rows(page, scholar_name, scholar_id, scholar_link):
        key = (rec["scholar_id"], rec["year"], rec["title"])
        if key in seen:
            continue
        seen.add(key)
        records.append(rec)

    records.sort(key=lambda r: (-int(r["year"]), str(r["title"]).lower()))
    return scholar_name, scholar_id, records


def main() -> None:
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=PROFILE_DIR,
            headless=HEADLESS,
            viewport={"width": 1400, "height": 900},
            locale="en-US",
        )
        page = context.new_page()

        try:
            scholar_name, scholar_id, records = scrape_one_profile(page, SCHOLAR_URL)
        finally:
            context.close()

    safe_name = safe_filename_name(scholar_name)
    output_csv = output_dir / f"{safe_name}_publications_{START_YEAR}_{END_YEAR}.csv"

    out_df = pd.DataFrame(
        records,
        columns=["scholar_name", "scholar_id", "google_scholar_link", "year", "title"],
    )

    if not out_df.empty:
        out_df = out_df.sort_values(
            ["year", "title"],
            ascending=[False, True],
        )

    out_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    # Also write a stable fallback filename, useful for automation/debugging.
    out_df.to_csv(DEFAULT_OUTPUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"\nSaved {len(out_df)} publication rows to: {output_csv}")
    print(f"Also saved a copy to: {DEFAULT_OUTPUT_CSV}")

    if len(out_df) == 0:
        print("\nNo 2024-2026 publications found or scraped.")
        print("If the profile clearly has papers, Google Scholar may have blocked the browser.")


if __name__ == "__main__":
    main()
