#!/usr/bin/env python3
"""
viterbi_hdbscan_centroid_label.py

Keyword-based department research landscape clustering.

Features:
- Generates SentenceTransformer embeddings from the 20 research keywords for each professor in a department
- Performs HDBSCAN clustering directly on the SentenceTransformer embeddings
- Support optional outlier reassignment
- Support optional keep noise as cluster

Cluster naming method:
- Primary cluster labels are generated using the top three representative research keywords, selected based on their proximity to the cluster centroid
- c-TF-IDF is also used to extract representative research keywords that capture the unique research characteristics of each professor/lab

Input file:
The input file (e.g., Book1_google_scholar.csv) should contain the following columns:
- Department
- PI/Lab, PI, Professor, scholar_name, name
- Research_Focus (used for clustering; keywords should be seperated by semicolon)
- PhD_Students, Phd_Students, PhD Students (student should be seperated by commas)

Outputs:
- Generates 15 different visulization files

Before running:
- Edit the CONFIG section as needed.
"""

import os
import re
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import plotly.express as px
import plotly.graph_objects as go

try:
    import umap
except Exception:
    umap = None

try:
    import hdbscan
except Exception as e:
    raise ImportError("Install required packages with: pip install hdbscan umap-learn sentence-transformers plotly kaleido openpyxl") from e


# ============================================================
# CONFIG. Change the paths and department name for each department
# ============================================================
INPUT_PATH = "./Book1_google_scholar.csv"
OUTDIR = "./output_google scholar approach"
DEPARTMENT_NAME = "Ming Hsieh Department of Electrical and Computer Engineering"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_KEYWORDS_PER_PI = 20
TOP_CLUSTER_KEYWORDS = 12
TOP_KEYWORDS_FOR_LEGEND = 3

# Number of original centroid-closest Research_Focus keywords used as primary clustering labels/legends.
TOP_CENTROID_KEYWORDS_FOR_CHART_LABEL = 3

# Cluster-label quality control
# A keyword must appear in at least this many PIs inside the cluster to be used 
# as a representative cluster keyword. This prevents one-off PI-specific keywords from dominating mixed clusters
MIN_PI_KEYWORD_DF_IN_CLUSTER = 2
FALLBACK_TO_SINGLE_PI_KEYWORDS = True  # if no shared keywords exist, still show fallback but mark cluster as mixed

# Old-script style HDBSCAN settings
MIN_CLUSTER_SIZE = 2
MIN_SAMPLES = 1
# good with normalized embeddings
HDBSCAN_METRIC = "euclidean" 
# "eom" or "leaf"        
CLUSTER_SELECTION_METHOD = "eom"     
CLUSTER_SELECTION_EPSILON = 0.0
ASSIGN_OUTLIERS = False
KEEP_NOISE_AS_CLUSTER = False

UMAP_METRIC = "cosine"
RANDOM_STATE = 0

GLOBAL_COLOR_SEQUENCE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#31a354", "#756bb1", "#636363", "#e6550d"
]


# ============================================================
# Helpers
# ============================================================
def safe_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    return s[:120]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).strip().lower())


def norm_text(t: str) -> str:
    t = str(t).replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def pct(x, total):
    return 100.0 * float(x) / float(total) if total else 0.0


def read_input(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("INPUT_PATH must be .xlsx, .xls, or .csv")
    df.columns = df.columns.astype(str).str.strip()
    return df


def find_column(df: pd.DataFrame, candidates, required=True, label="column"):
    norm_to_original = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = norm_col(cand)
        if key in norm_to_original:
            return norm_to_original[key]
    if required:
        raise ValueError(f"Could not find {label}. Tried {candidates}. Available columns: {list(df.columns)}")
    return None


def parse_keywords(cell, top_k=20):
    """Parse semicolon-separated keywords from the exact Research_Focus column.

    No conversion or phrase extraction is performed. The function only:
    1) splits on semicolons;
    2) strips extra spaces,
    3) removes duplicate keywords if any
    """
    if pd.isna(cell):
        return []

    parts = [x.strip() for x in str(cell).split(";") if x.strip()]

    out, seen = [], set()
    for kw in parts:
        low = kw.lower()
        if low not in seen:
            seen.add(low)
            out.append(kw)
        if len(out) >= top_k:
            break
    return out


def count_phd_students(cell) -> int:
    if pd.isna(cell):
        return 0
    s = str(cell).strip()
    if not s:
        return 0
    emails = re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", s)
    if emails:
        return len(set(e.lower() for e in emails))
    entries = [x.strip() for x in re.split(r"[;\n]+|,(?=\s*[A-Z][A-Za-z])", s) if x.strip()]
    return len(entries)


def remap_labels_by_size(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    counts = pd.Series(labels).value_counts().to_dict()
    ordered_old = sorted(np.unique(labels).tolist(), key=lambda x: (-counts[int(x)], int(x)))
    old_to_new = {int(old): i + 1 for i, old in enumerate(ordered_old)}
    return np.array([old_to_new[int(x)] for x in labels], dtype=int)


def make_theme_order(cluster_name_map):
    return [cluster_name_map[c] for c in sorted(cluster_name_map)]


def make_theme_color_map(cluster_name_map):
    return {
        cluster_name_map[c]: GLOBAL_COLOR_SEQUENCE[(i % len(GLOBAL_COLOR_SEQUENCE))]
        for i, c in enumerate(sorted(cluster_name_map))
    }


def try_write_plotly_png(fig, out_png: str, scale: int = 2):
    try:
        fig.write_image(out_png, scale=scale)
    except Exception as e:
        print(f"[WARN] Could not write PNG '{out_png}'. Install/upgrade kaleido if needed. {e}")


# ============================================================
# Embeddings + HDBSCAN
# ============================================================
def embed(texts, model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb)


def fit_hdbscan(embeddings):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(MIN_CLUSTER_SIZE),
        min_samples=None if MIN_SAMPLES is None else int(MIN_SAMPLES),
        metric=HDBSCAN_METRIC,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        cluster_selection_epsilon=float(CLUSTER_SELECTION_EPSILON),
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings)
    return clusterer, np.asarray(labels)


def assign_outliers_to_nearest_cluster(embeddings, labels):
    labels = np.asarray(labels).copy()
    non_outlier_mask = labels != -1
    if np.all(non_outlier_mask) or not np.any(non_outlier_mask):
        return labels

    uniq = sorted(set(int(x) for x in labels if int(x) != -1))
    centroids = []
    for c in uniq:
        idx = np.where(labels == c)[0]
        centroid = embeddings[idx].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        centroids.append(centroid)

    outlier_idx = np.where(labels == -1)[0]
    sims = cosine_similarity(embeddings[outlier_idx], np.vstack(centroids))
    nearest = sims.argmax(axis=1)
    labels[outlier_idx] = np.array([uniq[i] for i in nearest], dtype=int)
    return labels


def make_umap_coords(emb):
    if umap is None:
        raise RuntimeError("umap-learn not installed. Install with: pip install umap-learn")
    reducer = umap.UMAP(
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    return reducer.fit_transform(emb)

# ============================================================
# Embedding-centroid keyword centrality labeling
# ============================================================
def compute_keyword_centrality(labels, keyword_lists, professor_embeddings, model_name=MODEL_NAME, top_n=20):
    """Rank actual Research_Focus keywords by closeness to each cluster embedding centroid.

    This does NOT create generated label candidates, subphrases, or n-grams.
    It only uses the original parsed Top20 Research_Focus keywords for each professor/PI.
    """
    labels = np.asarray(labels)
    clusters = sorted(set(int(x) for x in labels))
    model = SentenceTransformer(model_name)

    rows = []
    top_map = {}

    for c in clusters:
        idx = np.where(labels == c)[0]
        centroid = professor_embeddings[idx].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        # Preserve the first observed original capitalization, but dedupe case-insensitively.
        kw_original = {}
        kw_pi_df = Counter()
        for i in idx:
            seen_this_pi = set()
            for kw in keyword_lists[i]:
                clean = norm_text(kw)
                low = clean.lower()
                if not low:
                    continue
                kw_original.setdefault(low, clean)
                if low not in seen_this_pi:
                    kw_pi_df[low] += 1
                    seen_this_pi.add(low)

        if not kw_original:
            top_map[c] = []
            continue

        lows = list(kw_original.keys())
        originals = [kw_original[x] for x in lows]
        kw_emb = model.encode(originals, show_progress_bar=False, normalize_embeddings=True)
        sims = cosine_similarity(centroid.reshape(1, -1), kw_emb).ravel()

        scored = []
        for low, orig, sim in zip(lows, originals, sims):
            scored.append((orig, float(sim), int(kw_pi_df[low])))

        scored.sort(key=lambda x: (-x[1], -x[2], x[0].lower()))
        top_map[c] = [kw for kw, _, _ in scored[:top_n]]

        for rank, (kw, sim, pi_df) in enumerate(scored, start=1):
            rows.append({
                "cluster": c,
                "rank": rank,
                "keyword": kw,
                "centrality_similarity": sim,
                "pi_df_in_cluster": pi_df,
                "cluster_size": int(len(idx)),
            })

    return top_map, pd.DataFrame(rows)


def build_cluster_name_map_from_keyword_centrality(labels, centrality_keyword_map, top_n=TOP_CENTROID_KEYWORDS_FOR_CHART_LABEL):
    """Legend/chart name = C# + top original keywords by centroid centrality."""
    name_map = {}
    for c in sorted(set(int(x) for x in labels)):
        kws = centrality_keyword_map.get(c, [])[:top_n]
        name_map[c] = f"C{c}: " + ", ".join(kws) if kws else f"C{c}"
    return name_map


# ============================================================
# Keyword-level c-TF-IDF labeling
# ============================================================
def compute_keyword_ctfidf(labels, keyword_lists, top_n=TOP_CLUSTER_KEYWORDS):
    """Rank cluster keywords using keyword-level c-TF-IDF
    To represent the unique characters for each professor in each specific cluster."""
    labels = np.asarray(labels)
    clusters = sorted(set(int(x) for x in labels))
    n_clusters = len(clusters)

    cluster_counts = {}
    cluster_total = {}
    cluster_pi_df = {}
    clusters_containing = defaultdict(int)

    # Build keyword counts and PI-document-frequency inside each cluster.
    for c in clusters:
        idx = np.where(labels == c)[0]
        counter = Counter()
        pi_df_counter = Counter()

        for i in idx:
            # Count each keyword at most once per PI so duplicate text inside a lab does not dominate.
            unique_kws = []
            seen = set()
            for kw in keyword_lists[i]:
                low = str(kw).strip().lower()
                if low and low not in seen:
                    seen.add(low)
                    unique_kws.append(low)

            counter.update(unique_kws)
            pi_df_counter.update(unique_kws)

        cluster_counts[c] = counter
        cluster_total[c] = sum(counter.values())
        cluster_pi_df[c] = pi_df_counter

        for kw in counter.keys():
            clusters_containing[kw] += 1

    rows = []
    top_map = {}
    quality_map = {}

    for c in clusters:
        scored_all = []
        scored_shared = []
        total = cluster_total[c]
        cluster_size = int(np.sum(labels == c))

        for kw, cnt in cluster_counts[c].items():
            tf = cnt / total if total else 0.0
            idf = math.log((1 + n_clusters) / (1 + clusters_containing[kw])) + 1.0
            score = tf * idf
            pi_df = int(cluster_pi_df[c].get(kw, 0))

            row = {
                "cluster": c,
                "keyword": kw,
                "ctfidf_score": score,
                "cluster_count": cnt,
                "pi_df_in_cluster": pi_df,
                "cluster_size": cluster_size,
                "clusters_containing_keyword": clusters_containing[kw],
                "tf": tf,
                "idf": idf,
                "passes_pi_df_filter": pi_df >= MIN_PI_KEYWORD_DF_IN_CLUSTER,
            }
            rows.append(row)

            item = (kw, score, cnt, pi_df, clusters_containing[kw], tf, idf)
            scored_all.append(item)
            if pi_df >= MIN_PI_KEYWORD_DF_IN_CLUSTER:
                scored_shared.append(item)

        scored_all.sort(key=lambda x: (-x[1], -x[3], -x[2], x[0]))
        scored_shared.sort(key=lambda x: (-x[1], -x[3], -x[2], x[0]))

        if scored_shared:
            chosen = scored_shared[:top_n]
            quality_map[c] = "shared-keyword"
        elif FALLBACK_TO_SINGLE_PI_KEYWORDS:
            chosen = scored_all[:top_n]
            quality_map[c] = "mixed-no-shared-keywords"
        else:
            chosen = []
            quality_map[c] = "mixed-no-shared-keywords"

        top_map[c] = [kw for kw, *_ in chosen]

    scores_df = pd.DataFrame(rows)
    if not scores_df.empty:
        scores_df = scores_df.sort_values(["cluster", "passes_pi_df_filter", "ctfidf_score"], ascending=[True, False, False])
    return top_map, scores_df, quality_map

def build_cluster_name_map(labels, top_keyword_map, top_for_name=TOP_KEYWORDS_FOR_LEGEND):
    name_map = {}
    for c in sorted(set(int(x) for x in labels)):
        kws = top_keyword_map.get(c, [])
        name_map[c] = f"C{c}: " + ", ".join(kws[:top_for_name]) if kws else f"C{c}"
    return name_map


# ============================================================
# Plots / tables
# ============================================================
def plot_theme_distribution_bars(labels, cluster_name_map, theme_color_map, out_png):
    s = pd.Series(labels).value_counts()
    clusters = sorted([int(c) for c in s.index])
    counts = np.array([int(s[c]) for c in clusters], dtype=int)
    total = int(counts.sum())

    names_full = [cluster_name_map[c] for c in clusters]
    names_short = [n if len(n) <= 55 else n[:52] + "..." for n in names_full]
    colors = [theme_color_map[cluster_name_map[c]] for c in clusters]

    x = np.arange(len(clusters))
    plt.figure(figsize=(14, 6))
    plt.bar(x, counts, color=colors)
    plt.xticks(x, names_short, rotation=25, ha="right")
    plt.ylabel("# PIs")
    plt.title("Department Landscape: Theme distribution (colored by theme)")
    for i, (v, p_) in enumerate(zip(counts, [pct(c, total) for c in counts])):
        plt.text(i, v + max(0.15, 0.01 * max(counts)), f"{p_:.1f}%", ha="center", fontsize=10)
    legend_handles = [Patch(facecolor=colors[i], label=names_full[i]) for i in range(len(clusters))]
    plt.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def plot_cluster_phd_distribution(cluster_stats, theme_color_map, out_png):
    dfp = cluster_stats.sort_values("cluster")
    names_full = dfp["Theme"].tolist()
    names_short = [n if len(n) <= 55 else n[:52] + "..." for n in names_full]
    vals = dfp["total_phd_students"].astype(int).tolist()
    colors = [theme_color_map[t] for t in names_full]

    x = np.arange(len(dfp))
    plt.figure(figsize=(14, 6))
    plt.bar(x, vals, color=colors)
    plt.xticks(x, names_short, rotation=25, ha="right")
    plt.ylabel("# PhD Students")
    plt.title("PhD Student Distribution Across Research Clusters")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.2, str(v), ha="center", fontsize=10)
    legend_handles = [Patch(facecolor=colors[i], label=names_full[i]) for i in range(len(dfp))]
    plt.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def plot_cluster_labs_vs_phd(cluster_stats, out_png):
    dfp = cluster_stats.sort_values("cluster")
    names = [n if len(n) <= 50 else n[:47] + "..." for n in dfp["Theme"].tolist()]
    labs = dfp["num_labs"].astype(int).tolist()
    phd = dfp["total_phd_students"].astype(int).tolist()

    x = np.arange(len(dfp))
    width = 0.38
    plt.figure(figsize=(15, 6))
    plt.bar(x - width / 2, labs, width=width, label="# Labs")
    plt.bar(x + width / 2, phd, width=width, label="# PhD Students")
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel("Count")
    plt.title("Research Clusters: Number of Labs vs Number of PhD Students")
    plt.legend()
    for i, v in enumerate(labs):
        plt.text(i - width / 2, v + 0.05, str(v), ha="center", fontsize=9)
    for i, v in enumerate(phd):
        plt.text(i + width / 2, v + 0.05, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def plot_cluster_top_keywords(top_keyword_map, cluster_name_map, out_png, centrality_keyword_map=None):
    """Create 03_theme_top_keywords PNG with two visually separated sections.

    Section 1: top 5 actual Research_Focus keywords closest to the cluster centroid.
    Section 2: top 5 actual Research_Focus keywords by keyword-level c-TF-IDF.

    The two sections use different text colors and lightly tinted boxes so the PNG
    is easier to read in presentations/reports.
    """
    centrality_keyword_map = centrality_keyword_map or {}
    clusters = sorted(set(top_keyword_map) | set(centrality_keyword_map))
    n = len(clusters)
    cols = 2 if n <= 6 else 3
    rows = int(np.ceil(n / cols)) if n else 1

    # Color choices for readability in the keyword PNG only.
    centrality_color = "#1f4e79"   # blue
    ctfidf_color = "#8a4b08"       # brown/orange
    centrality_box = "#eaf2fb"     # light blue background
    ctfidf_box = "#fff3e0"         # light orange background

    fig, axes = plt.subplots(rows, cols, figsize=(16, 6.4 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for ax in axes[n:]:
        ax.axis("off")

    for i, c in enumerate(clusters):
        ax = axes[i]
        ax.axis("off")
        title = cluster_name_map.get(c, f"C{c}")
        ax.set_title("\n".join(re.findall(r".{1,45}(?:\s+|$)", title)), fontsize=13, fontweight="bold", pad=10)

        centrality_lines = [f"• {kw}" for kw in centrality_keyword_map.get(c, [])[:5]] or ["• N/A"]
        ctfidf_lines = [f"• {kw}" for kw in top_keyword_map.get(c, [])[:5]] or ["• N/A"]

        ax.text(
            0.02, 0.96,
            "Top 5 by centroid centrality",
            va="top", ha="left", fontsize=12.2, fontweight="bold", color=centrality_color,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=centrality_box, edgecolor=centrality_color, linewidth=1.0),
            transform=ax.transAxes,
        )
        ax.text(
            0.04, 0.82,
            "\n".join(centrality_lines),
            va="top", ha="left", fontsize=11.4, color=centrality_color, linespacing=1.3,
            transform=ax.transAxes,
        )

        ax.text(
            0.02, 0.46,
            "Top 5 by keyword-level c-TF-IDF",
            va="top", ha="left", fontsize=12.2, fontweight="bold", color=ctfidf_color,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=ctfidf_box, edgecolor=ctfidf_color, linewidth=1.0),
            transform=ax.transAxes,
        )
        ax.text(
            0.04, 0.32,
            "\n".join(ctfidf_lines),
            va="top", ha="left", fontsize=11.4, color=ctfidf_color, linespacing=1.3,
            transform=ax.transAxes,
        )

    fig.suptitle("Theme top keywords: centroid centrality vs keyword-level c-TF-IDF", fontsize=18, y=1.02)
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)

def plot_keyword_heatmap(labels, keyword_lists, cluster_name_map, out_html, top_k=30):
    # Use top keywords from each cluster for vocabulary.
    top_map, _, _ = compute_keyword_ctfidf(labels, keyword_lists, top_n=8)
    vocab = []
    for c in sorted(top_map):
        vocab.extend(top_map[c])
    vocab = list(dict.fromkeys(vocab))[:top_k]

    y_labels = []
    mat = []
    for c in sorted(set(int(x) for x in labels)):
        idx = np.where(np.asarray(labels) == c)[0]
        counter = Counter()
        for i in idx:
            counter.update([str(k).strip().lower() for k in keyword_lists[i] if str(k).strip()])
        mat.append([counter.get(w, 0) for w in vocab])
        y_labels.append(cluster_name_map.get(c, f"C{c}"))

    fig = go.Figure(data=go.Heatmap(
        z=np.array(mat), x=vocab, y=y_labels,
        hovertemplate="Theme=%{y}<br>Keyword=%{x}<br>Count=%{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Theme Evidence: Cluster × Keyword Heatmap",
        xaxis_title="Keywords", yaxis_title="Themes",
        margin=dict(l=220, r=20, t=60, b=160),
        height=max(520, 45 * len(y_labels)),
    )
    fig.write_html(out_html)


def save_cluster_stats_table(cluster_stats, out_html):
    dfp = cluster_stats.copy()
    dfp["avg_phd_per_lab"] = dfp["avg_phd_per_lab"].round(2)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(dfp.columns)),
        cells=dict(values=[dfp[c].tolist() for c in dfp.columns])
    )])
    fig.update_layout(title="Cluster-level PhD Student Summary", margin=dict(l=10, r=10, t=60, b=10), height=500 + 30 * len(dfp))
    fig.write_html(out_html)


def plot_pi_cluster_table(pi_names, labels, keyword_texts, keyword_lists, cluster_name_map, emb, phd_counts, out_html):
    rep = np.zeros(len(pi_names), dtype=float)
    for c in sorted(set(int(x) for x in labels)):
        idx = np.where(labels == c)[0]
        centroid = emb[idx].mean(axis=0, keepdims=True)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        rep[idx] = cosine_similarity(emb[idx], centroid).ravel()

    df = pd.DataFrame({
        "PI": pi_names,
        "Cluster ID": [int(x) for x in labels],
        "Theme": [cluster_name_map[int(x)] for x in labels],
        "PhD Count": phd_counts,
        "Centrality": np.round(rep, 3),
        "Top20 Research Keywords": ["; ".join(kws) for kws in keyword_lists],
        "Research Focus (truncated)": [t[:260] + ("..." if len(t) > 260 else "") for t in keyword_texts],
    }).sort_values(["Cluster ID", "Centrality"], ascending=[True, False])

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns)),
        cells=dict(values=[df[c].tolist() for c in df.columns])
    )])
    fig.update_layout(title="PI → Theme / Cluster Mapping", margin=dict(l=10, r=10, t=60, b=10), height=900)
    fig.write_html(out_html)
    return df


def plot_umap_hover(pi_names, labels, coords, cluster_name_map, keyword_lists, phd_counts, out_html, out_png, theme_color_map, theme_order):
    theme = [cluster_name_map[int(l)] for l in labels]
    dfp = pd.DataFrame({
        "proj1": coords[:, 0],
        "proj2": coords[:, 1],
        "PI": pi_names,
        "Theme": theme,
        "Top Keywords": [", ".join(k[:5]) for k in keyword_lists],
        "PhD Count": phd_counts,
    })
    dfp["MarkerSymbol"] = np.where(dfp["PhD Count"].astype(int) > 0, "circle", "triangle-up")
    dfp["Plot Size"] = dfp["PhD Count"].astype(int).clip(lower=1)

    fig = px.scatter(
        dfp, x="proj1", y="proj2", color="Theme",
        color_discrete_map=theme_color_map,
        category_orders={"Theme": theme_order},
        size="Plot Size", hover_name="PI",
        hover_data={"Theme": True, "Top Keywords": True, "PhD Count": True, "MarkerSymbol": False, "Plot Size": False, "proj1": False, "proj2": False},
        title="Lab Landscape (UMAP projection) — each marker = one PI; color = theme; triangle = no PhD student found",
        labels={"proj1": "Embedding projection (dim 1) [UMAP]", "proj2": "Embedding projection (dim 2) [UMAP]"},
    )
    for trace in fig.data:
        mask = dfp["Theme"] == trace.name
        trace.marker.symbol = dfp.loc[mask, "MarkerSymbol"].tolist()
    fig.update_traces(marker=dict(line=dict(width=0.6, color="black")), selector=dict(mode="markers"))
    fig.write_html(out_html)
    try_write_plotly_png(fig, out_png, scale=2)


# ============================================================
# Main
# ============================================================
def main():
    out = os.path.join(OUTDIR, safe_name(DEPARTMENT_NAME))
    ensure_dir(out)

    df = read_input(INPUT_PATH)
    dept_col = find_column(df, ["Department", "department"], required=True, label="Department")
    pi_col = find_column(df, ["PI/Lab", "PI", "Professor", "Professor Name", "scholar_name", "name"], required=True, label="PI/Lab")
    focus_col = find_column(df, ["Research_Focus"], required=True, label="Research_Focus")
    phd_col = find_column(df, ["PhD_Students", "Phd_Students", "PhD Students", "phd_students", "Phd Students"], required=False, label="PhD_Students")

    sub = df[df[dept_col].astype(str).str.strip() == DEPARTMENT_NAME].copy()
    if len(sub) == 0:
        raise ValueError(f"No rows found where {dept_col} == '{DEPARTMENT_NAME}'. Available: {sorted(df[dept_col].dropna().astype(str).unique().tolist())}")
    sub = sub.dropna(subset=[pi_col, focus_col])

    pi_names = sub[pi_col].astype(str).tolist()
    keyword_texts = [norm_text(x) for x in sub[focus_col].astype(str).tolist()]
    # Embeddings use the Research_Focus string directly.
    keyword_lists = [parse_keywords(x, TOP_K_KEYWORDS_PER_PI) for x in sub[focus_col].tolist()]
    phd_counts = sub[phd_col].apply(count_phd_students).astype(int).tolist() if phd_col else [0] * len(sub)

    emb = embed(keyword_texts, MODEL_NAME)
    clusterer, raw_labels = fit_hdbscan(emb)

    labels_working = raw_labels.copy()
    if ASSIGN_OUTLIERS:
        labels_working = assign_outliers_to_nearest_cluster(emb, labels_working)

    if np.any(labels_working == -1):
        if KEEP_NOISE_AS_CLUSTER:
            labels_working = np.where(labels_working == -1, labels_working.min() - 1, labels_working)
        else:
            keep_idx = np.where(labels_working != -1)[0]
            if keep_idx.size == 0:
                raise ValueError("HDBSCAN labeled every point as noise. Try smaller MIN_CLUSTER_SIZE or set KEEP_NOISE_AS_CLUSTER=True.")
            print(f"[INFO] Dropped {len(labels_working) - len(keep_idx)} noise points.")
            pi_names = [pi_names[i] for i in keep_idx]
            keyword_texts = [keyword_texts[i] for i in keep_idx]
            keyword_lists = [keyword_lists[i] for i in keep_idx]
            phd_counts = [phd_counts[i] for i in keep_idx]
            emb = emb[keep_idx]
            labels_working = labels_working[keep_idx]

    labels = remap_labels_by_size(labels_working)

    # Supporting keywords: keyword-level c-TF-IDF. These are evidence, not the primary label.
    top_keyword_map, keyword_scores_df, cluster_quality_map = compute_keyword_ctfidf(labels, keyword_lists, TOP_CLUSTER_KEYWORDS)

    # Primary label: top 5 original Research_Focus keywords closest to each cluster centroid.
    # No generated label candidates, subphrases, or n-grams are used.
    centrality_keyword_map, keyword_centrality_df = compute_keyword_centrality(
        labels, keyword_lists, emb, MODEL_NAME, top_n=20
    )
    cluster_name_map = build_cluster_name_map_from_keyword_centrality(
        labels, centrality_keyword_map, TOP_CENTROID_KEYWORDS_FOR_CHART_LABEL    )

    theme_order = make_theme_order(cluster_name_map)
    theme_color_map = make_theme_color_map(cluster_name_map)

    cluster_df = pd.DataFrame({
        "Department": DEPARTMENT_NAME,
        "PI": pi_names,
        "cluster": labels,
        "hdbscan_label": labels_working,
        "phd_count": phd_counts,
        "Top20 Research Keywords": ["; ".join(kws) for kws in keyword_lists],
    })

    cluster_stats = (
        cluster_df.groupby("cluster")
        .agg(
            num_labs=("PI", "count"),
            total_phd_students=("phd_count", "sum"),
            avg_phd_per_lab=("phd_count", "mean"),
            median_phd_per_lab=("phd_count", "median"),
            max_phd_in_lab=("phd_count", "max"),
            labs=("PI", lambda x: "; ".join(map(str, x))),
        )
        .reset_index()
    )
    cluster_stats["Theme"] = cluster_stats["cluster"].map(cluster_name_map)
    cluster_stats["Top Centrality Keywords"] = cluster_stats["cluster"].map(lambda c: "; ".join(centrality_keyword_map.get(int(c), [])))
    cluster_stats["Label Quality"] = cluster_stats["cluster"].map(lambda c: cluster_quality_map.get(int(c), "unknown"))
    cluster_stats["Supporting c-TF-IDF Keywords"] = cluster_stats["cluster"].map(lambda c: "; ".join(top_keyword_map.get(int(c), [])))
    cluster_stats = cluster_stats[["cluster", "Theme", "Top Centrality Keywords", "Label Quality", "Supporting c-TF-IDF Keywords", "num_labs", "total_phd_students", "avg_phd_per_lab", "median_phd_per_lab", "max_phd_in_lab", "labs"]].sort_values("cluster")

    print("\nDepartment landscape:")
    s = pd.Series(labels).value_counts().sort_index()
    total = int(s.sum())
    for c, cnt in s.items():
        print(f"  {cluster_name_map[int(c)]} -> {int(cnt)} PIs ({pct(cnt, total):.1f}%)")

    plot_theme_distribution_bars(labels, cluster_name_map, theme_color_map, os.path.join(out, "01_theme_distribution_bars.png"))
    plot_cluster_phd_distribution(cluster_stats, theme_color_map, os.path.join(out, "02_cluster_phd_distribution.png"))
    plot_cluster_labs_vs_phd(cluster_stats, os.path.join(out, "02b_cluster_labs_vs_phd.png"))
    plot_cluster_top_keywords(top_keyword_map, cluster_name_map, os.path.join(out, "03_theme_top_keywords_centroid_and_ctfidf.png"), centrality_keyword_map)
    plot_keyword_heatmap(labels, keyword_lists, cluster_name_map, os.path.join(out, "04_theme_keyword_heatmap.html"))
    save_cluster_stats_table(cluster_stats, os.path.join(out, "04b_cluster_phd_summary_table.html"))
    pi_table = plot_pi_cluster_table(pi_names, labels, keyword_texts, keyword_lists, cluster_name_map, emb, phd_counts, os.path.join(out, "05_pi_cluster_table.html"))

    cluster_df.to_csv(os.path.join(out, "06_hdbscan_cluster_info.csv"), index=False)
    cluster_stats.to_csv(os.path.join(out, "01_cluster_summary_ctfidf_keywords.csv"), index=False)
    pi_table.to_csv(os.path.join(out, "02_professor_cluster_table.csv"), index=False)
    keyword_scores_df.to_csv(os.path.join(out, "03_keyword_ctfidf_scores_long.csv"), index=False)
    keyword_centrality_df.to_csv(os.path.join(out, "03b_keyword_centrality_scores_long.csv"), index=False)

    if umap is not None and len(emb) >= 3:
        coords = make_umap_coords(emb)
        plot_umap_hover(pi_names, labels, coords, cluster_name_map, keyword_lists, phd_counts,
                        os.path.join(out, "07_umap_hover.html"),
                        os.path.join(out, "07_umap_hover.png"),
                        theme_color_map, theme_order)
    else:
        print("[WARN] Skipping UMAP because umap-learn is unavailable or not enough points.")

    # One Excel workbook for convenient inspection. This is an output, not an input.
    with pd.ExcelWriter(os.path.join(out, "08_viterbi_ame_hdbscan_ctfidf_results.xlsx"), engine="openpyxl") as writer:
        cluster_stats.to_excel(writer, sheet_name="cluster_summary", index=False)
        pi_table.to_excel(writer, sheet_name="professor_clusters", index=False)
        keyword_scores_df.to_excel(writer, sheet_name="keyword_ctfidf_scores", index=False)
        keyword_centrality_df.to_excel(writer, sheet_name="keyword_centrality", index=False)
        cluster_df.to_excel(writer, sheet_name="hdbscan_info", index=False)

    print(f"\nSaved outputs to: {out}")
    print(" - 01_cluster_summary_ctfidf_keywords.csv")
    print(" - 01_theme_distribution_bars.png")
    print(" - 02_cluster_phd_distribution.png")
    print(" - 02_professor_cluster_table.csv")
    print(" - 02b_cluster_labs_vs_phd.png")
    print(" - 03_keyword_ctfidf_scores_long.csv")
    print(" - 03_theme_top_keywords_centroid_and_ctfidf.png")
    print(" - 03b_keyword_centrality_scores_long.csv")
    print(" - 04_theme_keyword_heatmap.html")
    print(" - 04b_cluster_phd_summary_table.html")
    print(" - 05_pi_cluster_table.html")
    print(" - 06_hdbscan_cluster_info.csv")
    print(" - 07_umap_hover.html / .png")
    print(" - 08_viterbi_ame_hdbscan_ctfidf_results.xlsx")


if __name__ == "__main__":
    main()
