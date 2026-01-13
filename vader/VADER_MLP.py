# =====================================================
# MLP VOLATILITY FORECASTING WITH AND WITHOUT SENTIMENT
# =====================================================

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# =====================================================
# CONFIGURATION
# =====================================================
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

processed_dir = BASE_DIR / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

REDDIT_CLEAN_FILE = BASE_DIR / "data" / "raw" / "Reddit_2021.csv"
BTC_FILE = BASE_DIR / "data" / "raw" / "btc_5min_2021.csv"

# =====================================================
# 1. LOAD REDDIT DATA
# =====================================================
print("Loading Reddit data...")
reddit = pd.read_csv(REDDIT_CLEAN_FILE)

def fix_datetime(dt):
    dt = str(dt)
    return f"{dt[:10]} {dt[10:12]}:00:00"

reddit["hour"] = reddit["hour"].apply(fix_datetime)
reddit["hour"] = pd.to_datetime(reddit["hour"], utc=True, errors="coerce")
reddit = reddit.dropna(subset=["hour"])
reddit["hour"] = reddit["hour"].dt.floor("h")

# =====================================================
# 2. APPLY VADER SENTIMENT
# =====================================================
print("Applying VADER sentiment...")
analyzer = SentimentIntensityAnalyzer()

reddit["vader"] = reddit["clean_text"].apply(
    lambda x: analyzer.polarity_scores(str(x))["compound"]
)

# =====================================================
# 3. HOURLY SENTIMENT AGGREGATION
# =====================================================
hourly_vader = (
    reddit
    .groupby("hour")
    .agg(
        sentiment_mean=("vader", "mean"),
        sentiment_std=("vader", "std"),
        volume=("vader", "count")
    )
    .reset_index()
)

# =====================================================
# 4. LOAD BTC DATA
# =====================================================
print("Loading BTC data...")
btc = pd.read_csv(BTC_FILE, parse_dates=["datetime"])
btc = btc.rename(columns={"datetime": "hour"})

# Ensure hour is timezone-aware (UTC)
if btc["hour"].dt.tz is None:
    btc["hour"] = btc["hour"].dt.tz_localize("UTC")
else:
    btc["hour"] = btc["hour"].dt.tz_convert("UTC")
btc = btc.sort_values("hour")

btc["log_price"] = np.log(btc["close"])
btc["ret"] = btc["log_price"].diff()

# =====================================================
# 5. MERGE BTC + SENTIMENT
# =====================================================
df = pd.merge(btc, hourly_vader, on="hour", how="inner")
df = df.dropna()

# =====================================================
# 6. DAILY REALISED VARIANCE (INTRADAY-BASED)
# =====================================================
# Compute daily realized variance as sum of squared intraday returns
# Using hourly returns as intraday frequency (since 5-minute bars not available)
df["date"] = df["hour"].dt.date

daily = (
    df.groupby("date")
    .agg(
        RV=("ret", lambda x: np.sum(x**2)),  # Realized variance = sum of squared returns
        sentiment=("sentiment_mean", "mean"),
        disagreement=("sentiment_std", "mean")
    )
    .reset_index()
)

# Calculate log of realized variance: lnRV_t = log(RV_t)
daily["lnRV"] = np.log(daily["RV"])

# =====================================================
# 7. HAR COMPONENTS (FROM INTRADAY-BASED lnRV)
# =====================================================
# Build HAR components using lnRV directly (not from volatility)
daily["lnRV_lag1"] = daily["lnRV"].shift(1)  # Daily lag
daily["lnRV_lag7"] = daily["lnRV"].rolling(7).mean().shift(1)  # Weekly average
daily["lnRV_lag30"] = daily["lnRV"].rolling(30).mean().shift(1)  # Monthly average

# Lag sentiment and disagreement by one day (day before)
daily["sentiment_lag1"] = daily["sentiment"].shift(1)
daily["disagreement_lag1"] = daily["disagreement"].shift(1)

daily = daily.dropna()

# =====================================================
# 8. CREATE 7-DAY-AHEAD TARGETS
# =====================================================
for k in range(1, 8):
    daily[f"lnRV_tplus{k}"] = daily["lnRV"].shift(-k)

daily = daily.dropna()

Y_cols = [f"lnRV_tplus{k}" for k in range(1, 8)]

# =====================================================
# 9. TRAIN / TEST SPLIT
# =====================================================
split = int(len(daily) * 0.7)
train = daily.iloc[:split]
test = daily.iloc[split:]

Y_train = train[Y_cols]
Y_test = test[Y_cols]

# =====================================================
# 10. BASELINE MLP (HAR ONLY)
# =====================================================
print("\nRunning baseline MLP (HAR inputs only)...")

X_base = ["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"]

scaler_base = StandardScaler()
X_train_base = scaler_base.fit_transform(train[X_base])
X_test_base = scaler_base.transform(test[X_base])

mlp_base = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation="tanh",
    solver="adam",
    max_iter=1000,
    tol=1e-4,
    random_state=42
)

mlp_base.fit(X_train_base, Y_train)
pred_base = mlp_base.predict(X_test_base)

mse_base = mean_squared_error(Y_test, pred_base)
print(f"Baseline MLP MSE: {mse_base:.6f}")

# Horizon-specific MSE for baseline MLP
print("\nHorizon-specific MSE (Baseline MLP):")
for i, col in enumerate(Y_cols):
    mse_h = mean_squared_error(Y_test.iloc[:, i], pred_base[:, i])
    print(f"t+{i+1}: {mse_h:.4f}")

# =====================================================
# 11. MLP + DICTIONARY SENTIMENT
# =====================================================
print("\nRunning MLP + dictionary-based sentiment...")

X_sent = [
    "lnRV_lag1",
    "lnRV_lag7",
    "lnRV_lag30",
    "sentiment_lag1",
    "disagreement_lag1"
]

scaler_sent = StandardScaler()
X_train_sent = scaler_sent.fit_transform(train[X_sent])
X_test_sent = scaler_sent.transform(test[X_sent])

mlp_sent = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation="tanh",
    solver="adam",
    max_iter=1000,
    tol=1e-4,
    random_state=42
)

mlp_sent.fit(X_train_sent, Y_train)
pred_sent = mlp_sent.predict(X_test_sent)

mse_sent = mean_squared_error(Y_test, pred_sent)
print(f"MLP + Sentiment MSE: {mse_sent:.6f}")

# =====================================================
# 12. HORIZON-SPECIFIC MSE (OPTIONAL)
# =====================================================
print("\nHorizon-specific MSE (MLP + Sentiment):")
for i, col in enumerate(Y_cols):
    mse_h = mean_squared_error(Y_test.iloc[:, i], pred_sent[:, i])
    print(f"t+{i+1}: {mse_h:.4f}")

# =====================================================
# 13. PLOT OUT-OF-SAMPLE: ACTUAL vs MLP vs MLP+SENT
# =====================================================

# Choose horizon to plot: 1..7
H = 1  # change to 2..7 if you want
h_idx = H - 1

# Build an out-of-sample results frame aligned to the TEST period
oos = test[["date"]].copy()
oos["date"] = pd.to_datetime(oos["date"])  # safe conversion (from python date)

# Actual future lnRV for horizon H (this is the correct target)
y_actual = Y_test.iloc[:, h_idx].values

# Predictions for the same horizon H
y_pred_base = pred_base[:, h_idx]
y_pred_sent = pred_sent[:, h_idx]

oos["actual_lnRV"] = y_actual
oos["pred_mlp_base"] = y_pred_base
oos["pred_mlp_sent"] = y_pred_sent

# Optional: compute BTC daily close to overlay on secondary axis
# (from your hourly btc dataframe already loaded earlier)
btc_daily_close = (
    btc.copy()
    .assign(date=btc["hour"].dt.date)
    .groupby("date")["close"]
    .last()
    .reset_index()
)
btc_daily_close["date"] = pd.to_datetime(btc_daily_close["date"])

oos = oos.merge(btc_daily_close, on="date", how="left")  # adds 'close'

# ---- Plot ----
fig, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(oos["date"], oos["actual_lnRV"], label=f"Actual lnRV (t+{H})")
ax1.plot(oos["date"], oos["pred_mlp_base"], label="MLP (HAR only)")
ax1.plot(oos["date"], oos["pred_mlp_sent"], label="MLP + Sentiment")

ax1.set_title(f"Out-of-sample forecasts: Actual vs MLP vs MLP+Sentiment (Horizon t+{H})")
ax1.set_xlabel("Date")
ax1.set_ylabel("Log Realized Variance (lnRV)")
ax1.legend(loc="upper left")
ax1.grid(True)

# Secondary axis for BTC price (optional but nice)
ax2 = ax1.twinx()
ax2.plot(oos["date"], oos["close"], label="BTC Daily Close", linestyle="--")
ax2.set_ylabel("BTC Daily Close Price")

# Separate legend for price axis
ax2.legend(loc="upper right")

plt.tight_layout()

# Save plot
plot_path = processed_dir / f"mlp_oos_vs_actual_h{H}.png"
plt.savefig(plot_path, dpi=200)
print(f"\nSaved plot to: {plot_path}")

plt.show()

print("\nMLP analysis complete.")
