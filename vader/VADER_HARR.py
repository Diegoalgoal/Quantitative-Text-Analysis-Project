import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =========================
# Configuration
# =========================
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

processed_dir = BASE_DIR / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Input files
REDDIT_CLEAN_FILE = BASE_DIR / "data" / "raw" / "Reddit_2021.csv"
BTC_FILE = BASE_DIR / "data" / "raw" / "btc_5min_2021.csv"

# =========================
# 1. LOAD AND PROCESS REDDIT DATA
# =========================
print("Loading cleaned Reddit data...")
reddit = pd.read_csv(REDDIT_CLEAN_FILE)

# Parse timestamp (format: YYYY-MM-DDHH, e.g., "2021-01-0100")
def fix_datetime_format(dt_str):
    """Fix datetime format YYYY-MM-DDHH to YYYY-MM-DD HH:00:00"""
    if pd.isna(dt_str):
        return dt_str
    dt_str = str(dt_str)
    if len(dt_str) >= 11 and dt_str[4] == "-" and dt_str[7] == "-":
        return f"{dt_str[:10]} {dt_str[10:12]}:00:00"
    return dt_str

reddit["hour"] = reddit["hour"].apply(fix_datetime_format)
reddit["hour"] = pd.to_datetime(reddit["hour"], format="%Y-%m-%d %H:%M:%S", utc=True, errors="coerce")

# Drop rows where datetime parsing failed
initial_count = len(reddit)
reddit = reddit.dropna(subset=["hour"])
dropped_count = initial_count - len(reddit)
if dropped_count > 0:
    print(f"Warning: Dropped {dropped_count} rows with invalid datetime values")

# Floor to hourly bins
reddit["hour"] = reddit["hour"].dt.floor("H")

# =========================
# 2. APPLY VADER SENTIMENT
# =========================
print("Applying VADER sentiment analysis...")
analyzer = SentimentIntensityAnalyzer()

reddit["vader_compound"] = reddit["clean_text"].apply(
    lambda x: analyzer.polarity_scores(str(x))["compound"] if pd.notna(x) else np.nan
)

# =========================
# 3. AGGREGATE TO HOURLY VADER INDEX
# =========================
print("Aggregating to hourly sentiment index...")

hourly_vader = (
    reddit
    .groupby("hour")
    .agg(
        vader_sentiment_mean=("vader_compound", "mean"),
        vader_sentiment_std=("vader_compound", "std"),
        comment_volume=("vader_compound", "count")
    )
    .reset_index()
)

print(f"Hourly VADER data: {len(hourly_vader):,} hours")

# =========================
# 4. LOAD BTC DATA
# =========================
print("Loading BTC hourly data...")
btc = pd.read_csv(BTC_FILE, parse_dates=["datetime"])

btc = btc.rename(columns={"datetime": "hour"})
btc = btc.sort_values("hour")

# Ensure hour is timezone-aware (UTC)
if btc["hour"].dt.tz is None:
    btc["hour"] = btc["hour"].dt.tz_localize("UTC")
else:
    btc["hour"] = btc["hour"].dt.tz_convert("UTC")

# =========================
# 5. CONSTRUCT RETURNS
# =========================
btc["log_price"] = np.log(btc["close"])
btc["ret"] = btc["log_price"].diff()

# =========================
# 6. MERGE BTC + VADER
# =========================
print("Merging BTC and VADER data...")
df = pd.merge(btc, hourly_vader, on="hour", how="inner")
df = df.dropna()

print(f"Merged data: {len(df):,} hourly observations")

# =========================
# 7. DAILY REALISED VOLATILITY
# =========================
df["date"] = df["hour"].dt.date

daily = (
    df.groupby("date")
    .agg(
        rv=("ret", lambda x: np.sqrt(np.sum(x**2))),
        sentiment=("vader_sentiment_mean", "mean"),
        disagreement=("vader_sentiment_std", "mean"),
    )
    .reset_index()
)

# Calculate realized variance (RV = rv^2)
daily["rv_squared"] = daily["rv"] ** 2

print(f"Daily data: {len(daily):,} days")

# =========================
# 8. HAR COMPONENTS (LOG TRANSFORMATIONS)
# =========================
# Calculate lagged realized volatilities
daily["rv_lag1"] = daily["rv"].shift(1)
daily["rv_lag7"] = daily["rv"].rolling(7).mean().shift(1)
daily["rv_lag30"] = daily["rv"].rolling(30).mean().shift(1)  # Monthly volatility

# Lag sentiment and disagreement by one day (day before)
daily["sentiment_lag1"] = daily["sentiment"].shift(1)
daily["disagreement_lag1"] = daily["disagreement"].shift(1)

# Calculate log of realized variance for dependent variable: lnRV_d,t
daily["lnRV"] = np.log(daily["rv_squared"])

# Calculate logs of lagged realized variances for regressors
daily["lnRV_lag1"] = np.log(daily["rv_lag1"] ** 2)
daily["lnRV_lag7"] = np.log(daily["rv_lag7"] ** 2)
daily["lnRV_lag30"] = np.log(daily["rv_lag30"] ** 2)

daily = daily.dropna()

# =========================
# 9. BASELINE HAR-ROBUST MODEL (LOG SPECIFICATION)
# =========================
# Dependent variable: log of daily realized variance (lnRV_d,t)
# Regressors: logs of lagged daily, weekly, and monthly realized variances
# Using HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors
X_base = daily[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"]]
X_base = sm.add_constant(X_base)
y = daily["lnRV"]

har_base = sm.OLS(y, X_base).fit(cov_type='HAC', cov_kwds={'maxlags': 7})

print("\n===== BASELINE HAR-ROBUST =====")
print(har_base.summary())

# =========================
# 10. HAR-ROBUST + VADER MODEL (LOG SPECIFICATION)
# =========================
# Dependent variable: log of daily realized variance (lnRV_d,t)
# Regressors: logs of lagged realized variances + lagged sentiment variables (day before)
# Using HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors
X_sent = daily[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30", "sentiment_lag1", "disagreement_lag1"]]
X_sent = sm.add_constant(X_sent)

har_sent = sm.OLS(y, X_sent).fit(cov_type='HAC', cov_kwds={'maxlags': 7})

print("\n===== HAR-ROBUST + VADER =====")
print(har_sent.summary())

# =========================
# 11. OUT-OF-SAMPLE RMSE (HAR-ROBUST LOG SPECIFICATION)
# =========================
split = int(len(daily) * 0.7)

train = daily.iloc[:split]
test = daily.iloc[split:]

X_train = sm.add_constant(
    train[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30", "sentiment_lag1", "disagreement_lag1"]]
)
X_test = sm.add_constant(
    test[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30", "sentiment_lag1", "disagreement_lag1"]]
)

y_train = train["lnRV"]
y_test = test["lnRV"]

model_oos = sm.OLS(y_train, X_train).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
pred = model_oos.predict(X_test)

rmse = np.sqrt(np.mean((y_test - pred) ** 2))
print(f"\nOut-of-sample RMSE (HAR-ROBUST + VADER, log specification): {rmse:.6f}")

# =========================
# 12. SAVE HOURLY VADER DATA
# =========================
hourly_vader_output = processed_dir / "reddit_2021_vader_hourly.csv"
hourly_vader.to_csv(hourly_vader_output, index=False)
print(f"\nHourly VADER data saved to: {hourly_vader_output}")

print("\nAnalysis complete!")

