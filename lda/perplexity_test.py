# =====================================================
# PERPLEXITY TEST SCRIPT
# Test different K values for LDA topic modeling
# =====================================================
"""
Perplexity Test Script for LDA Topic Modeling

This script tests different numbers of topics (K) using held-out perplexity
to determine the optimal number of topics for the Reddit dataset.

Usage:
    python perplexity_test.py

Output:
    - perplexity_results.csv: Perplexity scores for each K value
    - perplexity_results.txt: Formatted results summary
"""

import pandas as pd
import numpy as np
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# =====================================================
# CONFIGURATION
# =====================================================
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

# Data file paths
REDDIT_FILE = BASE_DIR / "data" / "raw" / "Reddit_2021.csv"

# LDA Configuration
K_MIN = 5
K_MAX = 30
K_STEP = 5
k_grid = list(range(K_MIN, K_MAX + 1, K_STEP))

# For faster testing, use a sample (set to None to use all data)
USE_SAMPLE = True
SAMPLE_SIZE = 200_000  # Use 200k documents for perplexity testing

# Vectorizer settings
MAX_FEATURES = 50_000
MIN_DF = 10
MAX_DF = 0.7
NGRAM_RANGE = (1, 1)

# LDA settings
LDA_MAX_ITER = 20
LDA_LEARNING_METHOD = "batch"
RANDOM_STATE = 42

# Minimum words per document
MIN_WORDS = 5

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def fix_datetime_format(dt_str):
    """Fix datetime format YYYY-MM-DDHH to YYYY-MM-DD HH:00:00"""
    if pd.isna(dt_str):
        return dt_str
    dt_str = str(dt_str)
    if len(dt_str) >= 11 and dt_str[4] == "-" and dt_str[7] == "-":
        return f"{dt_str[:10]} {dt_str[10:12]}:00:00"
    return dt_str


def simple_clean(text: str) -> str:
    """
    Minimal cleaning:
    - lowercase
    - remove urls
    - keep letters/numbers/spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class PerplexityResult:
    k: int
    perplexity: float


def select_k_via_perplexity(
    texts: List[str],
    k_grid: List[int],
    random_state: int = 42,
    max_features: int = 50_000,
    min_df: int = 10,
    max_df: float = 0.7,
    ngram_range: Tuple[int, int] = (1, 1),
    lda_max_iter: int = 20,
    lda_learning_method: str = "batch",
) -> pd.DataFrame:
    """
    Tests different K values using held-out perplexity.
    Returns a DataFrame with K and perplexity scores.
    """
    print("  Cleaning texts...", flush=True)
    cleaned = [simple_clean(t) for t in texts]
    print(f"  Cleaned {len(cleaned):,} texts", flush=True)

    print("  Vectorizing texts (this may take a while for large datasets)...", flush=True)
    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
    )
    X = vectorizer.fit_transform(cleaned)
    print(f"  Vectorization complete: {X.shape[0]:,} documents × {X.shape[1]:,} features", flush=True)
    if X.shape[0] < 100:
        print(f"[WARN] Very small corpus (n={X.shape[0]}). Perplexity selection may be unstable.", file=sys.stderr)

    print("  Splitting into train/test sets (80/20)...", flush=True)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=random_state)
    print("  Ready to fit LDA models", flush=True)

    rows = []

    print(f"\nTesting {len(k_grid)} values of K: {k_grid}")
    print(f"Training set size: {X_train.shape[0]:,} documents, {X_train.shape[1]:,} features")
    print(f"Test set size: {X_test.shape[0]:,} documents")
    
    for i, k in enumerate(k_grid, 1):
        print(f"\n[{i}/{len(k_grid)}] Fitting LDA with K={k}...", flush=True)
        lda = LatentDirichletAllocation(
            n_components=k,
            random_state=random_state,
            learning_method=lda_learning_method,
            max_iter=lda_max_iter,
            evaluate_every=1,
            n_jobs=-1,
            verbose=1,
        )
        lda.fit(X_train)
        print(f"  Computing perplexity on test set...", flush=True)
        ppx = lda.perplexity(X_test)
        rows.append({"K": k, "heldout_perplexity": ppx})
        print(f"  [K={k}] held-out perplexity: {ppx:.2f}", flush=True)

    results_df = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
    return results_df


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    print("="*60)
    print("LDA PERPLEXITY TEST")
    print("="*60)
    
    # Load Reddit data
    print("\n1. Loading Reddit data...")
    if not REDDIT_FILE.exists():
        raise FileNotFoundError(f"Reddit data file not found: {REDDIT_FILE}")
    
    print(f"Loading from: {REDDIT_FILE}")
    reddit = pd.read_csv(REDDIT_FILE)
    print(f"  - Loaded {len(reddit):,} observations")
    
    # Parse timestamp
    print("\n2. Parsing timestamps...")
    reddit["hour"] = reddit["hour"].apply(fix_datetime_format)
    reddit["hour"] = pd.to_datetime(reddit["hour"], format="%Y-%m-%d %H:%M:%S", utc=True, errors="coerce")
    reddit = reddit.dropna(subset=["hour"])
    reddit["hour"] = reddit["hour"].dt.floor("h")
    reddit = reddit.sort_values("hour").reset_index(drop=True)
    
    # Prepare texts
    print("\n3. Preparing texts...")
    texts = reddit["clean_text"].astype(str).tolist()
    print(f"  - Total documents: {len(texts):,}")
    
    # Filter documents with < 5 words
    print(f"\n4. Filtering documents (excluding those with < {MIN_WORDS} words)...")
    texts_filtered = []
    for text in texts:
        word_count = len(text.split())
        if word_count >= MIN_WORDS:
            texts_filtered.append(text)
    
    texts = texts_filtered
    print(f"  - Documents after filtering: {len(texts):,}")
    print(f"  - Excluded: {len(reddit) - len(texts):,} documents with < {MIN_WORDS} words")
    
    # Sample if requested
    if USE_SAMPLE and len(texts) > SAMPLE_SIZE:
        print(f"\n5. Sampling {SAMPLE_SIZE:,} documents for perplexity testing...")
        np.random.seed(RANDOM_STATE)
        sample_indices = np.random.choice(len(texts), size=SAMPLE_SIZE, replace=False)
        texts = [texts[i] for i in sample_indices]
        print(f"  - Using {len(texts):,} documents for testing")
    else:
        print(f"\n5. Using all {len(texts):,} documents for testing...")
    
    # Run perplexity tests
    print("\n" + "="*60)
    print("6. RUNNING PERPLEXITY TESTS")
    print("="*60)
    print(f"Testing K values: {k_grid}")
    
    results_df = select_k_via_perplexity(
        texts=texts,
        k_grid=k_grid,
        random_state=RANDOM_STATE,
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
        lda_max_iter=LDA_MAX_ITER,
        lda_learning_method=LDA_LEARNING_METHOD,
    )
    
    # Find best K
    best_k = results_df.loc[results_df["heldout_perplexity"].idxmin(), "K"]
    best_perplexity = results_df.loc[results_df["heldout_perplexity"].idxmin(), "heldout_perplexity"]
    
    # Save results
    print("\n" + "="*60)
    print("7. SAVING RESULTS")
    print("="*60)
    
    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_file = output_dir / "perplexity_results.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"✅ Saved CSV results to: {csv_file}")
    
    # Save formatted text report
    txt_file = output_dir / "perplexity_results.txt"
    with open(txt_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("LDA PERPLEXITY TEST RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Configuration:\n")
        f.write(f"  - K values tested: {k_grid}\n")
        f.write(f"  - Documents used: {len(texts):,}\n")
        f.write(f"  - Max features: {MAX_FEATURES:,}\n")
        f.write(f"  - Min document frequency: {MIN_DF}\n")
        f.write(f"  - Max document frequency: {MAX_DF}\n")
        f.write(f"  - LDA max iterations: {LDA_MAX_ITER}\n")
        f.write(f"  - Learning method: {LDA_LEARNING_METHOD}\n\n")
        f.write("Results:\n")
        f.write("-"*60 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        f.write("="*60 + "\n")
        f.write(f"BEST K: {best_k} (perplexity: {best_perplexity:.2f})\n")
        f.write("="*60 + "\n")
    
    print(f"✅ Saved text report to: {txt_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print("\nPerplexity Results:")
    print(results_df.to_string(index=False))
    print(f"\n✅ Best K: {best_k} (perplexity: {best_perplexity:.2f})")
    print(f"\nResults saved to: {output_dir}")
    print("\nYou can now use K={} in your main LDA script.".format(best_k))

