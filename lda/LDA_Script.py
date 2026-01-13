# =====================================================
# LDA + (optional) Hawkes topic dynamics + topic-frequency index
# for BTC volatility regressions
# =====================================================
"""
LDA + (optional) Hawkes topic dynamics + topic-frequency index for BTC volatility regressions.

What it does:
1) Loads Reddit r/bitcoin documents (must include timestamp + text).
2) Preprocesses text (lowercase, stopword removal, basic token filtering).
3) Fits LDA and selects number of topics K via held-out perplexity.
4) Extracts topic-word distributions (beta) and document-topic proportions (theta).
5) Aggregates topic prevalence over time (daily by default).
6) (Optional) Fits a Hawkes process per topic to quantify clustering/"impact".
   - Uses `tick` if installed; otherwise it skips Hawkes cleanly.
7) Builds a topic frequency index (and can merge with a volatility series for regressions).
"""

import pandas as pd
import numpy as np
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple
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
BTC_FILE = BASE_DIR / "data" / "raw" / "btc_5min_2021.csv"

# Alternative BTC file location (if 2021 file doesn't exist)
BTC_FILE_ALT = BASE_DIR / "data" / "raw" / "btc_5min_2017_2021.csv"

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




def parse_timestamp(series: pd.Series) -> pd.Series:
    """
    Accepts unix seconds OR datetime strings.
    """
    # If it's numeric-ish, treat as unix seconds
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", utc=True, errors="coerce").dt.tz_convert(None)
    # Otherwise parse as datetime
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)


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


def top_words_per_topic(model: LatentDirichletAllocation, feature_names: np.ndarray, n_top_words: int = 12) -> List[List[str]]:
    """Extract top words for each topic"""
    topics = []
    for k, comp in enumerate(model.components_):
        top_idx = np.argsort(comp)[::-1][:n_top_words]
        topics.append(feature_names[top_idx].tolist())
    return topics


@dataclass
class LDASelectionResult:
    best_k: int
    best_model: LatentDirichletAllocation
    vectorizer: CountVectorizer
    X_full: "scipy.sparse.csr_matrix"  # type: ignore
    doc_topic: np.ndarray              # theta_hat (n_docs x K)
    topic_words: List[List[str]]
    perplexity_table: pd.DataFrame


# NOTE: The select_k_via_perplexity function has been moved to perplexity_test.py
# Run perplexity_test.py separately to determine the optimal K value
# Then update the K variable in section 4 of this script

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
) -> LDASelectionResult:
    """
    DEPRECATED: This function has been moved to perplexity_test.py
    Fits a single CountVectorizer, then trains LDA for each K on train split,
    evaluates held-out perplexity on test split, selects best K (lowest perplexity).
    
    Use perplexity_test.py instead for perplexity testing.
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
    best_k = None
    best_ppx = np.inf
    best_model = None

    print(f"Testing {len(k_grid)} values of K: {k_grid}")
    print(f"Training set size: {X_train.shape[0]:,} documents, {X_train.shape[1]:,} features")
    print(f"Test set size: {X_test.shape[0]:,} documents")
    
    for i, k in enumerate(k_grid, 1):
        print(f"\n[{i}/{len(k_grid)}] Fitting LDA with K={k}...", flush=True)
        lda = LatentDirichletAllocation(
            n_components=k,
            random_state=random_state,
            learning_method=lda_learning_method,
            max_iter=lda_max_iter,
            evaluate_every=1,  # Show progress every iteration
            n_jobs=-1,
            verbose=1,  # Enable verbose output
        )
        lda.fit(X_train)
        print(f"  Computing perplexity on test set...", flush=True)
        ppx = lda.perplexity(X_test)
        rows.append({"K": k, "heldout_perplexity": ppx})
        if ppx < best_ppx:
            best_ppx = ppx
            best_k = k
            best_model = lda
        print(f"  [K={k}] held-out perplexity: {ppx:.2f} (best so far: K={best_k}, ppx={best_ppx:.2f})", flush=True)

    assert best_model is not None and best_k is not None

    # Refit best model on full corpus for final theta/beta
    print(f"\nRefitting best model (K={best_k}) on full corpus...", flush=True)
    best_model.fit(X)
    print("  Refitting complete", flush=True)

    # theta_hat for each doc
    print("  Computing document-topic proportions...", flush=True)
    doc_topic = best_model.transform(X)
    # Normalise (should already sum to 1, but numeric safety)
    doc_topic = normalize(doc_topic, norm="l1", axis=1)
    print("  Extracting top words per topic...", flush=True)

    feature_names = np.array(vectorizer.get_feature_names_out())
    topic_words = top_words_per_topic(best_model, feature_names, n_top_words=12)

    perplexity_table = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)

    return LDASelectionResult(
        best_k=best_k,
        best_model=best_model,
        vectorizer=vectorizer,
        X_full=X,
        doc_topic=doc_topic,
        topic_words=topic_words,
        perplexity_table=perplexity_table,
    )


def aggregate_topic_prevalence(
    df: pd.DataFrame,
    doc_topic: np.ndarray,
    time_col: str,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Builds time series:
    - sum_theta_k,t = sum of topic proportions within period t (intensity proxy)
    - mean_theta_k,t = average topic proportion within period t
    - n_docs_t = number of docs in period t
    """
    out = df[[time_col]].copy()
    out["bucket"] = pd.to_datetime(out[time_col]).dt.floor(freq)
    K = doc_topic.shape[1]

    # Attach theta columns
    for k in range(K):
        out[f"theta_{k}"] = doc_topic[:, k]

    grouped = out.groupby("bucket", as_index=False)
    agg_dict = {f"theta_{k}": ["sum", "mean"] for k in range(K)}
    agg_dict["bucket"] = "first"
    agg_dict["theta_0"].append("count")  # for doc count

    tmp = grouped.agg(agg_dict)

    # Flatten columns
    cols = []
    for c in tmp.columns:
        if isinstance(c, tuple):
            if c[0] == "bucket":
                cols.append("date")
            elif c[1] == "count":
                cols.append("n_docs")
            else:
                cols.append(f"{c[0]}_{c[1]}")  # theta_k_sum or theta_k_mean
        else:
            cols.append(str(c))
    tmp.columns = cols

    # Ensure date is datetime
    tmp["date"] = pd.to_datetime(tmp["date"])
    return tmp.sort_values("date").reset_index(drop=True)


def build_topic_frequency_index(topic_ts: pd.DataFrame, mode: str = "sum_over_topics") -> pd.DataFrame:
    """
    A simple index based on topic occurrence frequency/intensity.
    Since LDA gives proportions, "frequency" is usually operationalised as:
      - total topic mass in period: sum_k sum_theta_k,t
      - or focus on a subset of topics (e.g., most "important" topic from Hawkes)
    """
    df = topic_ts.copy()
    sum_cols = [c for c in df.columns if c.endswith("_sum") and c.startswith("theta_")]

    if mode == "sum_over_topics":
        df["topic_freq_index"] = df[sum_cols].sum(axis=1)
    elif mode == "mean_over_topics":
        mean_cols = [c for c in df.columns if c.endswith("_mean") and c.startswith("theta_")]
        df["topic_freq_index"] = df[mean_cols].mean(axis=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Standardise (z-score) for regression convenience
    mu = df["topic_freq_index"].mean()
    sd = df["topic_freq_index"].std(ddof=0)
    df["topic_freq_index_z"] = (df["topic_freq_index"] - mu) / (sd if sd > 0 else 1.0)
    return df


def try_fit_hawkes_per_topic(
    df: pd.DataFrame,
    doc_topic: np.ndarray,
    time_col: str,
    out_dir: str,
    event_rule: str = "argmax",
    min_events: int = 50,
) -> Optional[pd.DataFrame]:
    """
    Fits a univariate Hawkes with exponential kernel per topic using `tick` if available.
    Returns a table of Hawkes parameters per topic.

    Event construction choices:
    - "argmax": each document contributes one event to its highest-probability topic
    - "threshold": each document contributes an event to topic k if theta_{d,k} >= threshold (see below)
      (not implemented here to keep it simple)
    """
    try:
        from tick.hawkes import HawkesExpKern
    except Exception:
        print("[INFO] `tick` not available. Skipping Hawkes fitting.", file=sys.stderr)
        return None

    times = pd.to_datetime(df[time_col]).astype("datetime64[ns]")
    t0 = times.min()
    # Convert to float seconds since start (tick expects floats)
    t_seconds = (times - t0).dt.total_seconds().to_numpy()

    K = doc_topic.shape[1]
    topic_events: List[np.ndarray] = []
    if event_rule == "argmax":
        assigned = np.argmax(doc_topic, axis=1)
        for k in range(K):
            ev = np.sort(t_seconds[assigned == k])
            topic_events.append(ev)
    else:
        raise ValueError("Only event_rule='argmax' is implemented.")

    rows = []
    for k, ev in enumerate(topic_events):
        if ev.size < min_events:
            rows.append({"topic": k, "n_events": int(ev.size), "hawkes_mu": np.nan, "hawkes_alpha": np.nan, "hawkes_beta": np.nan, "branching_ratio": np.nan})
            continue

        # tick expects a list of realizations; each realization is a numpy array of event times
        hawkes = HawkesExpKern(decays=1.0, penalty="none", solver="agd", max_iter=200)
        hawkes.fit([ev])

        # For univariate, these are scalars in 1x1 matrices/vectors
        mu = float(hawkes.baseline[0])
        alpha = float(hawkes.adjacency[0, 0])
        beta = float(hawkes.decays[0])  # exponential decay parameter
        # A common "impact" summary is branching ratio (expected offspring per event).
        # For exp kernel phi(t)=alpha*exp(-beta t), branching ratio = alpha/beta (univariate).
        br = alpha / beta if beta > 0 else np.nan

        rows.append({
            "topic": k,
            "n_events": int(ev.size),
            "hawkes_mu": mu,
            "hawkes_alpha": alpha,
            "hawkes_beta": beta,
            "branching_ratio": br,
        })

    hawkes_table = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    hawkes_table.to_csv(Path(out_dir) / "hawkes_params_per_topic.csv", index=False)
    return hawkes_table

# =====================================================
# 1. LOAD REDDIT DATA
# =====================================================
print("="*60)
print("1. LOADING REDDIT DATA")
print("="*60)

if not REDDIT_FILE.exists():
    raise FileNotFoundError(f"Reddit data file not found: {REDDIT_FILE}")

print(f"Loading Reddit data from: {REDDIT_FILE}")
reddit = pd.read_csv(REDDIT_FILE)

print(f"  - Shape: {reddit.shape}")
print(f"  - Columns: {reddit.columns.tolist()}")

# Parse timestamp (format: YYYY-MM-DDHH, e.g., "2021-01-0100")
print("\nParsing datetime column...")
reddit["hour"] = reddit["hour"].apply(fix_datetime_format)
reddit["hour"] = pd.to_datetime(reddit["hour"], format="%Y-%m-%d %H:%M:%S", utc=True, errors="coerce")
reddit = reddit.dropna(subset=["hour"])
reddit["hour"] = reddit["hour"].dt.floor("h")

# Sort by hour
reddit = reddit.sort_values("hour").reset_index(drop=True)

print(f"  - Loaded {len(reddit):,} observations")
print(f"  - Date range: {reddit['hour'].min()} to {reddit['hour'].max()}")
print(f"  - Years covered: {sorted(reddit['hour'].dt.year.unique())}")

# Display sample
print("\n  Sample Reddit data:")
print(reddit.head())

# =====================================================
# 2. LOAD BITCOIN PRICE DATA
# =====================================================
print("\n" + "="*60)
print("2. LOADING BITCOIN PRICE DATA")
print("="*60)

# Check which BTC file exists
if BTC_FILE.exists():
    btc_file_path = BTC_FILE
elif BTC_FILE_ALT.exists():
    btc_file_path = BTC_FILE_ALT
    print(f"  Note: Using alternative file: {BTC_FILE_ALT}")
else:
    raise FileNotFoundError(f"Bitcoin data file not found. Checked:\n  - {BTC_FILE}\n  - {BTC_FILE_ALT}")

print(f"Loading Bitcoin data from: {btc_file_path}")
btc = pd.read_csv(btc_file_path)

print(f"  - Shape: {btc.shape}")
print(f"  - Columns: {btc.columns.tolist()}")

# Parse datetime
print("\nParsing datetime column...")
btc["datetime"] = pd.to_datetime(btc["datetime"], errors="coerce")
btc = btc.dropna(subset=["datetime"])
btc = btc.sort_values("datetime").reset_index(drop=True)

# Compute log_return if not present
if "log_return" not in btc.columns:
    print("  Computing log returns...")
    btc["log_return"] = np.log(btc["close"] / btc["close"].shift(1))
    btc = btc.dropna()

print(f"  - Loaded {len(btc):,} observations")
print(f"  - Date range: {btc['datetime'].min()} to {btc['datetime'].max()}")
print(f"  - Years covered: {sorted(btc['datetime'].dt.year.unique())}")
print(f"  - Frequency: 5-minute intervals")

# Display basic statistics
print("\n  Bitcoin price statistics:")
print(f"    - Price range: ${btc['close'].min():.2f} - ${btc['close'].max():.2f}")
print(f"    - Mean price: ${btc['close'].mean():.2f}")
print(f"    - Mean log return: {btc['log_return'].mean():.6f}")
print(f"    - Std log return: {btc['log_return'].std():.6f}")

# Display sample
print("\n  Sample Bitcoin data:")
print(btc.head())

# =====================================================
# 3. SUMMARY
# =====================================================
print("\n" + "="*60)
print("3. DATA LOADING SUMMARY")
print("="*60)
print(f"✅ Reddit data: {len(reddit):,} observations")
print(f"✅ Bitcoin data: {len(btc):,} observations")
print("\nData loaded successfully and ready for analysis!")

# =====================================================
# 4. LDA TOPIC MODELING
# =====================================================
print("\n" + "="*60)
print("4. LDA TOPIC MODELING")
print("="*60)

# Configuration for LDA
# Set K to the optimal value determined from perplexity testing
# Run perplexity_test.py first to determine the best K value
K = 5  # TODO: Update this value after running perplexity_test.py

print(f"Using K = {K} topics")
print("(Run perplexity_test.py to determine optimal K)")

# Prepare texts from Reddit data
print("\nPreparing texts for LDA...")
texts = reddit["clean_text"].astype(str).tolist()
print(f"  - Total documents: {len(texts):,}")

# Filter out documents with less than 5 words
print("\nFiltering documents (excluding those with < 5 words)...")
min_words = 5
texts_filtered = []
indices_kept = []
for i, text in enumerate(texts):
    word_count = len(text.split())
    if word_count >= min_words:
        texts_filtered.append(text)
        indices_kept.append(i)

texts = texts_filtered
print(f"  - Documents after filtering: {len(texts):,} ({len(texts)/len(reddit)*100:.1f}% of original)")
print(f"  - Excluded: {len(reddit) - len(texts):,} documents with < {min_words} words")

# Also filter the reddit dataframe to match (needed for later aggregation)
reddit_filtered = reddit.iloc[indices_kept].reset_index(drop=True)

# Clean and vectorize texts
print("\nCleaning and vectorizing texts...", flush=True)
cleaned = [simple_clean(t) for t in texts]
print(f"  Cleaned {len(cleaned):,} texts", flush=True)

print("  Vectorizing texts (this may take a while for large datasets)...", flush=True)
vectorizer = CountVectorizer(
    stop_words="english",
    max_features=50_000,
    min_df=10,
    max_df=0.7,
    ngram_range=(1, 1),
)
X = vectorizer.fit_transform(cleaned)
print(f"  Vectorization complete: {X.shape[0]:,} documents × {X.shape[1]:,} features", flush=True)

# Fit LDA model with fixed K
print(f"\nFitting LDA model with K={K}...", flush=True)
lda_model = LatentDirichletAllocation(
    n_components=K,
    random_state=42,
    learning_method="batch",
    max_iter=20,
    evaluate_every=1,
    n_jobs=-1,
    verbose=1,
)
lda_model.fit(X)
print("  LDA fitting complete", flush=True)

# Compute document-topic proportions
print("  Computing document-topic proportions...", flush=True)
doc_topic = lda_model.transform(X)
doc_topic = normalize(doc_topic, norm="l1", axis=1)

# Extract top words per topic
print("  Extracting top words per topic...", flush=True)
feature_names = np.array(vectorizer.get_feature_names_out())
topic_words = top_words_per_topic(lda_model, feature_names, n_top_words=12)

print(f"\n✅ LDA model fitted with K={K}")

# Display top words for each topic
print("\n" + "="*60)
print("Top words per topic:")
print("="*60)
for k, words in enumerate(topic_words):
    print(f"\nTopic {k}: {', '.join(words[:10])}")

# =====================================================
# 5. AGGREGATE TOPIC PREVALENCE OVER TIME
# =====================================================
print("\n" + "="*60)
print("5. AGGREGATING TOPIC PREVALENCE OVER TIME")
print("="*60)

# Aggregate to daily frequency
print("Aggregating topic proportions to daily frequency...")
# Use filtered reddit dataframe that matches the texts used for LDA
topic_ts = aggregate_topic_prevalence(
    df=reddit_filtered,
    doc_topic=doc_topic,
    time_col="hour",
    freq="D",
)

print(f"  - Daily topic time series: {len(topic_ts):,} days")
print(f"  - Date range: {topic_ts['date'].min()} to {topic_ts['date'].max()}")
print(f"  - Columns: {topic_ts.columns.tolist()}")

# Build topic frequency index
print("\nBuilding topic frequency index...")
topic_ts = build_topic_frequency_index(topic_ts, mode="sum_over_topics")
print(f"  - Topic frequency index created")
print(f"  - Mean index: {topic_ts['topic_freq_index'].mean():.4f}")
print(f"  - Std index: {topic_ts['topic_freq_index'].std():.4f}")

# =====================================================
# 6. OPTIONAL: HAWKES PROCESS FITTING
# =====================================================
print("\n" + "="*60)
print("6. OPTIONAL: HAWKES PROCESS FITTING")
print("="*60)

output_dir = BASE_DIR / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

hawkes_table = try_fit_hawkes_per_topic(
    df=reddit_filtered,
    doc_topic=doc_topic,
    time_col="hour",
    out_dir=str(output_dir),
    event_rule="argmax",
    min_events=50,
)

if hawkes_table is not None:
    print("\nHawkes process parameters per topic:")
    print(hawkes_table.to_string(index=False))
else:
    print("\nHawkes fitting skipped (tick library not available)")

# =====================================================
# 7. SAVE RESULTS
# =====================================================
print("\n" + "="*60)
print("7. SAVING RESULTS")
print("="*60)

# Save topic time series
topic_ts_file = output_dir / "topic_time_series.csv"
topic_ts.to_csv(topic_ts_file, index=False)
print(f"✅ Saved topic time series to: {topic_ts_file}")

# Save topic words
topic_words_file = output_dir / "topic_words.txt"
with open(topic_words_file, "w") as f:
    for k, words in enumerate(topic_words):
        f.write(f"Topic {k}: {', '.join(words)}\n")
print(f"✅ Saved topic words to: {topic_words_file}")

# Save model and vectorizer (optional, for reuse)
import pickle
model_file = output_dir / "lda_model.pkl"
vectorizer_file = output_dir / "vectorizer.pkl"
with open(model_file, "wb") as f:
    pickle.dump(lda_model, f)
with open(vectorizer_file, "wb") as f:
    pickle.dump(vectorizer, f)
print(f"✅ Saved LDA model to: {model_file}")
print(f"✅ Saved vectorizer to: {vectorizer_file}")

# =====================================================
# 8. SUMMARY
# =====================================================
print("\n" + "="*60)
print("8. ANALYSIS SUMMARY")
print("="*60)
print(f"✅ Reddit data: {len(reddit):,} observations")
print(f"✅ Bitcoin data: {len(btc):,} observations")
print(f"✅ Topics (K): {K}")
print(f"✅ Topic time series: {len(topic_ts):,} days")
print("\nLDA analysis complete! Results saved to output directory.")
print("\nData available for further analysis:")
print("  - reddit: DataFrame with columns ['hour', 'clean_text']")
print("  - btc: DataFrame with columns ['datetime', 'open', 'high', 'low', 'close', 'volume', 'log_return']")
print("  - topic_ts: Daily topic prevalence time series")
print("  - lda_model: Fitted LDA model")
print("  - vectorizer: Fitted CountVectorizer")
print("  - doc_topic: Document-topic proportions (n_docs × K)")
print("  - topic_words: Top words for each topic")

