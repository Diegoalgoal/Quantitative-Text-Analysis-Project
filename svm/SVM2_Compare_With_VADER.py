# =====================================================
# COMBINED SVM2 + VADER SENTIMENT-BASED VOLATILITY FORECASTING
# HAR-R vs MLP vs HAR-R+VADER vs HAR-R+SVM vs MLP+VADER vs MLP+SVM Models
# =====================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =====================================================
# CONFIGURATION
# =====================================================
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

processed_dir = BASE_DIR / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Data directory
data_dir = BASE_DIR / "data" / "raw"

# Initial training window size (6 months ≈ 180 days)
# Will use expanding window: start with INITIAL_WINDOW_SIZE, then expand
INITIAL_WINDOW_SIZE = 180

# =====================================================
# 1. LOAD AND PROCESS SVM SENTIMENT DATA
# =====================================================
print("="*60)
print("1. LOADING SVM SENTIMENT DATA")
print("="*60)

sentiment_raw = pd.read_csv(
    data_dir / "reddit_sentiment_svm_full.csv"
)

sentiment_raw["date"] = pd.to_datetime(sentiment_raw["date"])
# Extract just the date part (without time)
sentiment_raw["date"] = sentiment_raw["date"].dt.date

print(f"Loaded {len(sentiment_raw):,} SVM sentiment observations")

# Aggregate by date to create daily sentiment metrics
print("Aggregating SVM sentiment to daily frequency...")
sentiment_svm = sentiment_raw.groupby("date").agg({
    "sentiment_continuous": ["mean", "std"],  # mean and std for sentiment
    "disagreement_tweet": "mean",  # mean disagreement
    "date": "count"  # count of posts per day
}).reset_index()

# Flatten column names
sentiment_svm.columns = ["date", "sentiment_mean", "sentiment_dispersion", "disagreement", "n_posts"]

# Fill NaN values in sentiment_dispersion (occurs when only one post per day)
sentiment_svm["sentiment_dispersion"] = sentiment_svm["sentiment_dispersion"].fillna(0)

print(f"Aggregated to {len(sentiment_svm):,} days")
print(f"Date range: {sentiment_svm['date'].min()} to {sentiment_svm['date'].max()}")

# =====================================================
# 2. LOAD AND PROCESS VADER SENTIMENT DATA
# =====================================================
print("\n" + "="*60)
print("2. LOADING VADER SENTIMENT DATA")
print("="*60)

# Check for hourly VADER data
vader_hourly_file = processed_dir / "reddit_2021_vader_hourly.csv"
reddit_clean_file = data_dir / "Reddit_2021.csv"

if vader_hourly_file.exists():
    print(f"Loading pre-computed hourly VADER data from {vader_hourly_file}...")
    hourly_vader = pd.read_csv(vader_hourly_file, parse_dates=["hour"])
    if hourly_vader["hour"].dt.tz is None:
        hourly_vader["hour"] = pd.to_datetime(hourly_vader["hour"]).dt.tz_localize("UTC")
    else:
        hourly_vader["hour"] = pd.to_datetime(hourly_vader["hour"]).dt.tz_convert("UTC")
elif reddit_clean_file.exists():
    print(f"Computing VADER sentiment from {reddit_clean_file}...")
    reddit = pd.read_csv(reddit_clean_file)
    
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
    reddit = reddit.dropna(subset=["hour"])
    reddit["hour"] = reddit["hour"].dt.floor("h")
    
    # Filter to 2021 only
    reddit = reddit[reddit["hour"].dt.year == 2021].copy()
    print(f"Reddit data filtered to 2021: {len(reddit):,} observations")
    
    # Apply VADER sentiment
    print("Applying VADER sentiment analysis...")
    analyzer = SentimentIntensityAnalyzer()
    
    reddit["vader_compound"] = reddit["clean_text"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"] if pd.notna(x) else np.nan
    )
    
    # Aggregate to hourly VADER index
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
else:
    raise FileNotFoundError(f"Neither VADER hourly file ({vader_hourly_file}) nor Reddit cleaned file ({reddit_clean_file}) found")

# Aggregate VADER to daily
print("Aggregating VADER sentiment to daily frequency...")
daily_vader = (
    hourly_vader.groupby(hourly_vader["hour"].dt.date)
    .agg(
        sentiment_mean=("vader_sentiment_mean", "mean"),
        sentiment_dispersion=("vader_sentiment_std", "mean"),
        disagreement=("vader_sentiment_std", "mean"),  # Use std as disagreement measure
        n_posts=("comment_volume", "sum")
    )
    .reset_index()
)
daily_vader.rename(columns={"hour": "date"}, inplace=True)

print(f"Aggregated to {len(daily_vader):,} days")
print(f"Date range: {daily_vader['date'].min()} to {daily_vader['date'].max()}")

# =====================================================
# 3. LOAD BTC DATA (5-MINUTE)
# =====================================================
print("\n" + "="*60)
print("3. LOADING BTC DATA (5-MINUTE)")
print("="*60)

btc_file = data_dir / "btc_5min_2021.csv"
if not btc_file.exists():
    raise FileNotFoundError(f"BTC 5-minute data file not found: {btc_file}")

print("Loading BTC 5-minute data...")
btc = pd.read_csv(btc_file)
btc["datetime"] = pd.to_datetime(btc["datetime"])
btc = btc.sort_values("datetime")

# Filter to 2021 only (matching Reddit data)
print("Filtering to calendar year 2021 only...")
btc = btc[btc["datetime"].dt.year == 2021].copy()
print(f"BTC 5-minute data for 2021: {len(btc):,} observations")

# Use log_return if available, otherwise compute it
if "log_return" not in btc.columns:
    btc["log_return"] = np.log(btc["close"] / btc["close"].shift(1))
btc = btc.dropna()

print(f"BTC 5-minute data: {len(btc):,} observations")

# =====================================================
# 4. CONSTRUCT DAILY REALIZED VARIANCE FROM 5-MINUTE RETURNS
# =====================================================
print("\n" + "="*60)
print("4. CONSTRUCTING DAILY REALIZED VARIANCE")
print("="*60)
print("Computing daily RV from 5-minute returns...")
btc["date"] = btc["datetime"].dt.date

# Daily realized variance = sum of squared 5-minute returns
daily_rv = (
    btc.groupby("date")
    .agg(
        RV=("log_return", lambda x: np.sum(x**2)),  # Realized variance
    )
    .reset_index()
)

print(f"Daily RV computed: {len(daily_rv):,} days")
print(f"Date range: {daily_rv['date'].min()} to {daily_rv['date'].max()}")

# =====================================================
# 5. MERGE BTC RV + SVM SENTIMENT + VADER SENTIMENT (DAILY)
# =====================================================
print("\n" + "="*60)
print("5. MERGING BTC DAILY RV WITH SVM AND VADER SENTIMENT DATA")
print("="*60)

# Merge daily RV with daily SVM sentiment
daily = pd.merge(daily_rv, sentiment_svm, on="date", how="inner")
daily = daily.rename(columns={
    "sentiment_mean": "sentiment_svm_mean",
    "sentiment_dispersion": "sentiment_svm_dispersion",
    "disagreement": "disagreement_svm",
    "n_posts": "n_posts_svm"
})

# Merge with VADER sentiment
daily = pd.merge(daily, daily_vader, on="date", how="inner")
daily = daily.rename(columns={
    "sentiment_mean": "sentiment_vader_mean",
    "sentiment_dispersion": "sentiment_vader_dispersion",
    "disagreement": "disagreement_vader",
    "n_posts": "n_posts_vader"
})

daily = daily.dropna()

# Calculate log of realized variance
daily["lnRV"] = np.log(daily["RV"])

# Calculate realized volatility (sqrt of RV) for lagged components
daily["rv"] = np.sqrt(daily["RV"])

# Calculate lagged realized volatilities
daily["rv_lag1"] = daily["rv"].shift(1)
daily["rv_lag7"] = daily["rv"].rolling(7).mean().shift(1)
daily["rv_lag30"] = daily["rv"].rolling(30).mean().shift(1)

# Calculate logs of lagged realized variances for regressors
daily["lnRV_lag1"] = np.log(daily["rv_lag1"] ** 2)
daily["lnRV_lag7"] = np.log(daily["rv_lag7"] ** 2)
daily["lnRV_lag30"] = np.log(daily["rv_lag30"] ** 2)

# Lag sentiment and disagreement by one day (both SVM and VADER)
daily["sentiment_svm_lag1"] = daily["sentiment_svm_mean"].shift(1)
daily["disagreement_svm_lag1"] = daily["disagreement_svm"].shift(1)
daily["sentiment_vader_lag1"] = daily["sentiment_vader_mean"].shift(1)
daily["disagreement_vader_lag1"] = daily["disagreement_vader"].shift(1)

daily = daily.dropna()

# Ensure we only have 2021 data
daily["date"] = pd.to_datetime(daily["date"])
daily = daily[daily["date"].dt.year == 2021].copy()
daily = daily.sort_values("date").reset_index(drop=True)

print(f"Merged data: {len(daily):,} days")
print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")
print(f"Note: Analysis restricted to calendar year 2021 only")

# =====================================================
# 6. IN-SAMPLE MODEL ESTIMATION AND DIAGNOSTICS
# =====================================================
print("\n" + "="*60)
print("6. IN-SAMPLE MODEL ESTIMATION")
print("="*60)

# Use initial training window for in-sample diagnostics
train_end_idx = INITIAL_WINDOW_SIZE
if train_end_idx >= len(daily):
    raise ValueError(f"Initial window size ({INITIAL_WINDOW_SIZE}) exceeds available data ({len(daily)} days)")

train_data = daily.iloc[:train_end_idx].copy()
print(f"Using first {len(train_data)} days for in-sample estimation")
print(f"Training period: {train_data['date'].min()} to {train_data['date'].max()}")

# Prepare data for in-sample estimation
y = train_data["lnRV"].values

# Base features (HAR-R and MLP baseline)
X_base = train_data[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"]].values
X_base_df = pd.DataFrame(X_base, columns=["ln(RV_daily_t-1)", "ln(RV_weekly_t-1)", "ln(RV_monthly_t-1)"])
X_base_const = sm.add_constant(X_base_df)

# HAR-R + VADER features
X_vader = train_data[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30", "sentiment_vader_lag1", "disagreement_vader_lag1"]].values
X_vader_df = pd.DataFrame(X_vader, columns=["ln(RV_daily_t-1)", "ln(RV_weekly_t-1)", "ln(RV_monthly_t-1)", "vader_sentiment_t-1", "vader_disagreement_t-1"])
X_vader_const = sm.add_constant(X_vader_df)

# HAR-R + SVM features
X_svm = train_data[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30", "sentiment_svm_lag1", "disagreement_svm_lag1"]].values
X_svm_df = pd.DataFrame(X_svm, columns=["ln(RV_daily_t-1)", "ln(RV_weekly_t-1)", "ln(RV_monthly_t-1)", "svm_sentiment_t-1", "svm_disagreement_t-1"])
X_svm_const = sm.add_constant(X_svm_df)

# Scale features for MLP
scaler_base = StandardScaler()
scaler_vader = StandardScaler()
scaler_svm = StandardScaler()

X_base_scaled = scaler_base.fit_transform(X_base)
X_vader_scaled = scaler_vader.fit_transform(X_vader)
X_svm_scaled = scaler_svm.fit_transform(X_svm)

# Model 1: HAR-Robust (Baseline)
print("\n--- HAR-Robust (Baseline) ---")
har_model = sm.OLS(y, X_base_const).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
har_diagnostics = {
    "Model": "HAR-R",
    "N": har_model.nobs,
    "R²": har_model.rsquared,
    "Adj R²": har_model.rsquared_adj,
    "F-statistic": har_model.fvalue,
    "F p-value": har_model.f_pvalue,
    "Log-likelihood": har_model.llf,
    "AIC": har_model.aic,
    "BIC": har_model.bic,
    "Residual variance": har_model.mse_resid
}
print(f"N: {har_diagnostics['N']}")
print(f"R²: {har_diagnostics['R²']:.6f}")
print(f"Adj R²: {har_diagnostics['Adj R²']:.6f}")
print(f"AIC: {har_diagnostics['AIC']:.6f}")
print(f"BIC: {har_diagnostics['BIC']:.6f}")

# Model 2: MLP (Baseline)
print("\n--- MLP (Baseline) ---")
mlp_base_model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation='tanh',
    solver='adam',
    max_iter=2000,
    tol=1e-4,
    random_state=42,
    early_stopping=False
)
mlp_base_model.fit(X_base_scaled, y)
mlp_base_pred = mlp_base_model.predict(X_base_scaled)

n_base = len(y)
ss_res_base = np.sum((y - mlp_base_pred) ** 2)
ss_tot_base = np.sum((y - np.mean(y)) ** 2)
r2_base = 1 - (ss_res_base / ss_tot_base)
r2_adj_base = 1 - (1 - r2_base) * (n_base - 1) / (n_base - X_base.shape[1] - 1)
mse_base = ss_res_base / n_base
log_likelihood_base = -0.5 * n_base * (np.log(2 * np.pi) + np.log(mse_base) + 1)
aic_base = -2 * log_likelihood_base + 2 * (X_base.shape[1] + 1)
bic_base = -2 * log_likelihood_base + np.log(n_base) * (X_base.shape[1] + 1)

mlp_base_diagnostics = {
    "Model": "MLP",
    "N": n_base,
    "R²": r2_base,
    "Adj R²": r2_adj_base,
    "F-statistic": np.nan,
    "F p-value": np.nan,
    "Log-likelihood": log_likelihood_base,
    "AIC": aic_base,
    "BIC": bic_base,
    "Residual variance": mse_base
}
print(f"N: {mlp_base_diagnostics['N']}")
print(f"R²: {mlp_base_diagnostics['R²']:.6f}")
print(f"Adj R²: {mlp_base_diagnostics['Adj R²']:.6f}")
print(f"AIC: {mlp_base_diagnostics['AIC']:.6f}")
print(f"BIC: {mlp_base_diagnostics['BIC']:.6f}")

# Model 3: HAR-Robust + VADER
print("\n--- HAR-Robust + VADER ---")
har_vader_model = sm.OLS(y, X_vader_const).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
har_vader_diagnostics = {
    "Model": "HAR-R + VADER",
    "N": har_vader_model.nobs,
    "R²": har_vader_model.rsquared,
    "Adj R²": har_vader_model.rsquared_adj,
    "F-statistic": har_vader_model.fvalue,
    "F p-value": har_vader_model.f_pvalue,
    "Log-likelihood": har_vader_model.llf,
    "AIC": har_vader_model.aic,
    "BIC": har_vader_model.bic,
    "Residual variance": har_vader_model.mse_resid
}
print(f"N: {har_vader_diagnostics['N']}")
print(f"R²: {har_vader_diagnostics['R²']:.6f}")
print(f"Adj R²: {har_vader_diagnostics['Adj R²']:.6f}")
print(f"AIC: {har_vader_diagnostics['AIC']:.6f}")
print(f"BIC: {har_vader_diagnostics['BIC']:.6f}")

# Model 4: HAR-Robust + SVM
print("\n--- HAR-Robust + SVM ---")
har_svm_model = sm.OLS(y, X_svm_const).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
har_svm_diagnostics = {
    "Model": "HAR-R + SVM",
    "N": har_svm_model.nobs,
    "R²": har_svm_model.rsquared,
    "Adj R²": har_svm_model.rsquared_adj,
    "F-statistic": har_svm_model.fvalue,
    "F p-value": har_svm_model.f_pvalue,
    "Log-likelihood": har_svm_model.llf,
    "AIC": har_svm_model.aic,
    "BIC": har_svm_model.bic,
    "Residual variance": har_svm_model.mse_resid
}
print(f"N: {har_svm_diagnostics['N']}")
print(f"R²: {har_svm_diagnostics['R²']:.6f}")
print(f"Adj R²: {har_svm_diagnostics['Adj R²']:.6f}")
print(f"AIC: {har_svm_diagnostics['AIC']:.6f}")
print(f"BIC: {har_svm_diagnostics['BIC']:.6f}")

# Model 5: MLP + VADER
print("\n--- MLP + VADER ---")
mlp_vader_model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation='tanh',
    solver='adam',
    max_iter=2000,
    tol=1e-4,
    random_state=42,
    early_stopping=False
)
mlp_vader_model.fit(X_vader_scaled, y)
mlp_vader_pred = mlp_vader_model.predict(X_vader_scaled)

n_vader = len(y)
ss_res_vader = np.sum((y - mlp_vader_pred) ** 2)
ss_tot_vader = np.sum((y - np.mean(y)) ** 2)
r2_vader = 1 - (ss_res_vader / ss_tot_vader)
r2_adj_vader = 1 - (1 - r2_vader) * (n_vader - 1) / (n_vader - X_vader.shape[1] - 1)
mse_vader = ss_res_vader / n_vader
log_likelihood_vader = -0.5 * n_vader * (np.log(2 * np.pi) + np.log(mse_vader) + 1)
aic_vader = -2 * log_likelihood_vader + 2 * (X_vader.shape[1] + 1)
bic_vader = -2 * log_likelihood_vader + np.log(n_vader) * (X_vader.shape[1] + 1)

mlp_vader_diagnostics = {
    "Model": "MLP + VADER",
    "N": n_vader,
    "R²": r2_vader,
    "Adj R²": r2_adj_vader,
    "F-statistic": np.nan,
    "F p-value": np.nan,
    "Log-likelihood": log_likelihood_vader,
    "AIC": aic_vader,
    "BIC": bic_vader,
    "Residual variance": mse_vader
}
print(f"N: {mlp_vader_diagnostics['N']}")
print(f"R²: {mlp_vader_diagnostics['R²']:.6f}")
print(f"Adj R²: {mlp_vader_diagnostics['Adj R²']:.6f}")
print(f"AIC: {mlp_vader_diagnostics['AIC']:.6f}")
print(f"BIC: {mlp_vader_diagnostics['BIC']:.6f}")

# Model 6: MLP + SVM
print("\n--- MLP + SVM ---")
mlp_svm_model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation='tanh',
    solver='adam',
    max_iter=2000,
    tol=1e-4,
    random_state=42,
    early_stopping=False
)
mlp_svm_model.fit(X_svm_scaled, y)
mlp_svm_pred = mlp_svm_model.predict(X_svm_scaled)

n_svm = len(y)
ss_res_svm = np.sum((y - mlp_svm_pred) ** 2)
ss_tot_svm = np.sum((y - np.mean(y)) ** 2)
r2_svm = 1 - (ss_res_svm / ss_tot_svm)
r2_adj_svm = 1 - (1 - r2_svm) * (n_svm - 1) / (n_svm - X_svm.shape[1] - 1)
mse_svm = ss_res_svm / n_svm
log_likelihood_svm = -0.5 * n_svm * (np.log(2 * np.pi) + np.log(mse_svm) + 1)
aic_svm = -2 * log_likelihood_svm + 2 * (X_svm.shape[1] + 1)
bic_svm = -2 * log_likelihood_svm + np.log(n_svm) * (X_svm.shape[1] + 1)

mlp_svm_diagnostics = {
    "Model": "MLP + SVM",
    "N": n_svm,
    "R²": r2_svm,
    "Adj R²": r2_adj_svm,
    "F-statistic": np.nan,
    "F p-value": np.nan,
    "Log-likelihood": log_likelihood_svm,
    "AIC": aic_svm,
    "BIC": bic_svm,
    "Residual variance": mse_svm
}
print(f"N: {mlp_svm_diagnostics['N']}")
print(f"R²: {mlp_svm_diagnostics['R²']:.6f}")
print(f"Adj R²: {mlp_svm_diagnostics['Adj R²']:.6f}")
print(f"AIC: {mlp_svm_diagnostics['AIC']:.6f}")
print(f"BIC: {mlp_svm_diagnostics['BIC']:.6f}")

# Save diagnostics to DataFrame
diagnostics_df = pd.DataFrame([
    har_diagnostics,
    mlp_base_diagnostics,
    har_vader_diagnostics,
    har_svm_diagnostics,
    mlp_vader_diagnostics,
    mlp_svm_diagnostics
])
diagnostics_output = processed_dir / "svm2_vader_combined_diagnostics.csv"
diagnostics_df.to_csv(diagnostics_output, index=False)
print(f"\n✅ In-sample diagnostics saved to: {diagnostics_output}")

# =====================================================
# 7. EXPANDING WINDOW OUT-OF-SAMPLE FORECASTING
# =====================================================
print("\n" + "="*60)
print("7. EXPANDING WINDOW OUT-OF-SAMPLE FORECASTING")
print("="*60)

# Start forecasting after initial window
start_idx = INITIAL_WINDOW_SIZE
if start_idx >= len(daily):
    raise ValueError(f"Initial window size ({INITIAL_WINDOW_SIZE}) exceeds available data ({len(daily)} days)")

print(f"Initial training window: {INITIAL_WINDOW_SIZE} days (~6 months)")
print(f"Total days available: {len(daily):,}")
print(f"Forecast days: {len(daily) - start_idx:,}")
print(f"Using EXPANDING window: training set grows as we forecast forward...")

# Storage for forecasts (in ln(RV) space)
har_forecasts_lnRV = []
mlp_base_forecasts_lnRV = []
har_vader_forecasts_lnRV = []
har_svm_forecasts_lnRV = []
mlp_vader_forecasts_lnRV = []
mlp_svm_forecasts_lnRV = []
actual_lnRV = []
forecast_dates = []

# Storage for residual variances (for back-transformation)
har_residual_vars = []
mlp_base_residual_vars = []
har_vader_residual_vars = []
har_svm_residual_vars = []
mlp_vader_residual_vars = []
mlp_svm_residual_vars = []

print("\nGenerating expanding window forecasts...")
for i in range(start_idx, len(daily)):
    if (i - start_idx) % 50 == 0:
        print(f"  Processing day {i - start_idx + 1}/{len(daily) - start_idx}...")
    
    # Expanding window: use all data from start up to (but not including) current day
    train_window = daily.iloc[:i].copy()
    test_row = daily.iloc[i:i+1].copy()
    
    # Prepare training data
    X_train_base = train_window[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"]].values
    X_train_vader = train_window[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30", 
                                  "sentiment_vader_lag1", "disagreement_vader_lag1"]].values
    X_train_svm = train_window[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30", 
                                 "sentiment_svm_lag1", "disagreement_svm_lag1"]].values
    y_train = train_window["lnRV"].values
    
    # Prepare test data
    X_test_base = test_row[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"]].values
    X_test_vader = test_row[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30", 
                              "sentiment_vader_lag1", "disagreement_vader_lag1"]].values
    X_test_svm = test_row[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30", 
                            "sentiment_svm_lag1", "disagreement_svm_lag1"]].values
    y_test = test_row["lnRV"].values[0]
    
    # Ensure test data is 2D (single row)
    if X_test_base.ndim == 1:
        X_test_base = X_test_base.reshape(1, -1)
    if X_test_vader.ndim == 1:
        X_test_vader = X_test_vader.reshape(1, -1)
    if X_test_svm.ndim == 1:
        X_test_svm = X_test_svm.reshape(1, -1)
    
    # Scale features for MLP (separate scalers for each model type)
    scaler_base = StandardScaler()
    scaler_vader = StandardScaler()
    scaler_svm = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    X_train_vader_scaled = scaler_vader.fit_transform(X_train_vader)
    X_train_svm_scaled = scaler_svm.fit_transform(X_train_svm)
    X_test_base_scaled = scaler_base.transform(X_test_base)
    X_test_vader_scaled = scaler_vader.transform(X_test_vader)
    X_test_svm_scaled = scaler_svm.transform(X_test_svm)
    
    # Add constant for OLS (ensure 2D arrays)
    X_train_base_const = sm.add_constant(X_train_base, has_constant='add')
    X_test_base_const = sm.add_constant(X_test_base, has_constant='add')
    X_train_vader_const = sm.add_constant(X_train_vader, has_constant='add')
    X_test_vader_const = sm.add_constant(X_test_vader, has_constant='add')
    X_train_svm_const = sm.add_constant(X_train_svm, has_constant='add')
    X_test_svm_const = sm.add_constant(X_test_svm, has_constant='add')
    
    # HAR-Robust forecast
    try:
        har_model_roll = sm.OLS(y_train, X_train_base_const).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
        har_pred_lnRV = har_model_roll.predict(X_test_base_const)[0]
        har_residual_var = har_model_roll.mse_resid
    except Exception as e:
        if (i - start_idx) < 5:
            print(f"    Warning: HAR-Robust forecast failed at day {i-start_idx+1}: {e}")
        har_pred_lnRV = np.nan
        har_residual_var = np.nan
    
    # MLP baseline forecast
    try:
        mlp_base_model_roll = MLPRegressor(
            hidden_layer_sizes=(10,),
            activation='tanh',
            solver='adam',
            max_iter=2000,
            tol=1e-4,
            random_state=42,
            early_stopping=False
        )
        mlp_base_model_roll.fit(X_train_base_scaled, y_train)
        mlp_base_pred_lnRV = mlp_base_model_roll.predict(X_test_base_scaled.reshape(1, -1))[0]
        mlp_base_train_pred = mlp_base_model_roll.predict(X_train_base_scaled)
        mlp_base_residual_var = np.mean((y_train - mlp_base_train_pred) ** 2)
    except Exception as e:
        if (i - start_idx) < 5:
            print(f"    Warning: MLP baseline forecast failed at day {i-start_idx+1}: {e}")
        mlp_base_pred_lnRV = np.nan
        mlp_base_residual_var = np.nan
    
    # HAR-Robust + VADER forecast
    try:
        har_vader_model_roll = sm.OLS(y_train, X_train_vader_const).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
        har_vader_pred_lnRV = har_vader_model_roll.predict(X_test_vader_const)[0]
        har_vader_residual_var = har_vader_model_roll.mse_resid
    except Exception as e:
        if (i - start_idx) < 5:
            print(f"    Warning: HAR-Robust+VADER forecast failed at day {i-start_idx+1}: {e}")
        har_vader_pred_lnRV = np.nan
        har_vader_residual_var = np.nan
    
    # HAR-Robust + SVM forecast
    try:
        har_svm_model_roll = sm.OLS(y_train, X_train_svm_const).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
        har_svm_pred_lnRV = har_svm_model_roll.predict(X_test_svm_const)[0]
        har_svm_residual_var = har_svm_model_roll.mse_resid
    except Exception as e:
        if (i - start_idx) < 5:
            print(f"    Warning: HAR-Robust+SVM forecast failed at day {i-start_idx+1}: {e}")
        har_svm_pred_lnRV = np.nan
        har_svm_residual_var = np.nan
    
    # MLP + VADER forecast
    try:
        mlp_vader_model_roll = MLPRegressor(
            hidden_layer_sizes=(10,),
            activation='tanh',
            solver='adam',
            max_iter=2000,
            tol=1e-4,
            random_state=42,
            early_stopping=False
        )
        mlp_vader_model_roll.fit(X_train_vader_scaled, y_train)
        mlp_vader_pred_lnRV = mlp_vader_model_roll.predict(X_test_vader_scaled.reshape(1, -1))[0]
        mlp_vader_train_pred = mlp_vader_model_roll.predict(X_train_vader_scaled)
        mlp_vader_residual_var = np.mean((y_train - mlp_vader_train_pred) ** 2)
    except Exception as e:
        if (i - start_idx) < 5:
            print(f"    Warning: MLP+VADER forecast failed at day {i-start_idx+1}: {e}")
        mlp_vader_pred_lnRV = np.nan
        mlp_vader_residual_var = np.nan
    
    # MLP + SVM forecast
    try:
        mlp_svm_model_roll = MLPRegressor(
            hidden_layer_sizes=(10,),
            activation='tanh',
            solver='adam',
            max_iter=2000,
            tol=1e-4,
            random_state=42,
            early_stopping=False
        )
        mlp_svm_model_roll.fit(X_train_svm_scaled, y_train)
        mlp_svm_pred_lnRV = mlp_svm_model_roll.predict(X_test_svm_scaled.reshape(1, -1))[0]
        mlp_svm_train_pred = mlp_svm_model_roll.predict(X_train_svm_scaled)
        mlp_svm_residual_var = np.mean((y_train - mlp_svm_train_pred) ** 2)
    except Exception as e:
        if (i - start_idx) < 5:
            print(f"    Warning: MLP+SVM forecast failed at day {i-start_idx+1}: {e}")
        mlp_svm_pred_lnRV = np.nan
        mlp_svm_residual_var = np.nan
    
    # Store forecasts (in ln(RV) space)
    har_forecasts_lnRV.append(har_pred_lnRV)
    mlp_base_forecasts_lnRV.append(mlp_base_pred_lnRV)
    har_vader_forecasts_lnRV.append(har_vader_pred_lnRV)
    har_svm_forecasts_lnRV.append(har_svm_pred_lnRV)
    mlp_vader_forecasts_lnRV.append(mlp_vader_pred_lnRV)
    mlp_svm_forecasts_lnRV.append(mlp_svm_pred_lnRV)
    actual_lnRV.append(y_test)
    forecast_dates.append(test_row["date"].values[0])
    
    # Store residual variances
    har_residual_vars.append(har_residual_var)
    mlp_base_residual_vars.append(mlp_base_residual_var)
    har_vader_residual_vars.append(har_vader_residual_var)
    har_svm_residual_vars.append(har_svm_residual_var)
    mlp_vader_residual_vars.append(mlp_vader_residual_var)
    mlp_svm_residual_vars.append(mlp_svm_residual_var)

# =====================================================
# 8. OUT-OF-SAMPLE MSE EVALUATION (ln(RV) SPACE)
# =====================================================
print("\n" + "="*60)
print("8. OUT-OF-SAMPLE FORECAST EVALUATION")
print("="*60)

# Convert to arrays
har_forecasts_lnRV = np.array(har_forecasts_lnRV)
mlp_base_forecasts_lnRV = np.array(mlp_base_forecasts_lnRV)
har_vader_forecasts_lnRV = np.array(har_vader_forecasts_lnRV)
har_svm_forecasts_lnRV = np.array(har_svm_forecasts_lnRV)
mlp_vader_forecasts_lnRV = np.array(mlp_vader_forecasts_lnRV)
mlp_svm_forecasts_lnRV = np.array(mlp_svm_forecasts_lnRV)
actual_lnRV = np.array(actual_lnRV)
har_residual_vars = np.array(har_residual_vars)
mlp_base_residual_vars = np.array(mlp_base_residual_vars)
har_vader_residual_vars = np.array(har_vader_residual_vars)
har_svm_residual_vars = np.array(har_svm_residual_vars)
mlp_vader_residual_vars = np.array(mlp_vader_residual_vars)
mlp_svm_residual_vars = np.array(mlp_svm_residual_vars)

# Filter out NaN predictions
har_valid = ~np.isnan(har_forecasts_lnRV)
mlp_base_valid = ~np.isnan(mlp_base_forecasts_lnRV)
har_vader_valid = ~np.isnan(har_vader_forecasts_lnRV)
har_svm_valid = ~np.isnan(har_svm_forecasts_lnRV)
mlp_vader_valid = ~np.isnan(mlp_vader_forecasts_lnRV)
mlp_svm_valid = ~np.isnan(mlp_svm_forecasts_lnRV)

# Calculate MSE in ln(RV) space
def calc_mse(actual, pred, valid_mask):
    if valid_mask.sum() > 0:
        return mean_squared_error(actual[valid_mask], pred[valid_mask])
    return np.nan

har_mse_lnRV = calc_mse(actual_lnRV, har_forecasts_lnRV, har_valid)
mlp_base_mse_lnRV = calc_mse(actual_lnRV, mlp_base_forecasts_lnRV, mlp_base_valid)
har_vader_mse_lnRV = calc_mse(actual_lnRV, har_vader_forecasts_lnRV, har_vader_valid)
har_svm_mse_lnRV = calc_mse(actual_lnRV, har_svm_forecasts_lnRV, har_svm_valid)
mlp_vader_mse_lnRV = calc_mse(actual_lnRV, mlp_vader_forecasts_lnRV, mlp_vader_valid)
mlp_svm_mse_lnRV = calc_mse(actual_lnRV, mlp_svm_forecasts_lnRV, mlp_svm_valid)

print("\nOut-of-Sample MSE (ln(RV) space):")
print(f"  HAR-R:                        {har_mse_lnRV:.8f} (N={har_valid.sum()})" if not np.isnan(har_mse_lnRV) else f"  HAR-R:                        N/A (no valid forecasts)")
print(f"  MLP:                           {mlp_base_mse_lnRV:.8f} (N={mlp_base_valid.sum()})" if not np.isnan(mlp_base_mse_lnRV) else f"  MLP:                           N/A (no valid forecasts)")
print(f"  HAR-R + VADER:                 {har_vader_mse_lnRV:.8f} (N={har_vader_valid.sum()})" if not np.isnan(har_vader_mse_lnRV) else f"  HAR-R + VADER:                 N/A (no valid forecasts)")
print(f"  HAR-R + SVM:                   {har_svm_mse_lnRV:.8f} (N={har_svm_valid.sum()})" if not np.isnan(har_svm_mse_lnRV) else f"  HAR-R + SVM:                   N/A (no valid forecasts)")
print(f"  MLP + VADER:                   {mlp_vader_mse_lnRV:.8f} (N={mlp_vader_valid.sum()})" if not np.isnan(mlp_vader_mse_lnRV) else f"  MLP + VADER:                   N/A (no valid forecasts)")
print(f"  MLP + SVM:                     {mlp_svm_mse_lnRV:.8f} (N={mlp_svm_valid.sum()})" if not np.isnan(mlp_svm_mse_lnRV) else f"  MLP + SVM:                     N/A (no valid forecasts)")

# Calculate MSE on RV scale (back-transformed)
print("\nOut-of-Sample MSE (RV space, back-transformed):")
actual_RV_forecast = daily.iloc[start_idx:]["RV"].values

def calc_mse_rv(forecasts_lnRV, residual_vars, actual_RV, valid_mask):
    if valid_mask.sum() > 0:
        forecasts_RV = np.exp(forecasts_lnRV[valid_mask] + 0.5 * residual_vars[valid_mask])
        actual_RV_valid = actual_RV[valid_mask]
        return mean_squared_error(actual_RV_valid, forecasts_RV)
    return np.nan

har_mse_RV = calc_mse_rv(har_forecasts_lnRV, har_residual_vars, actual_RV_forecast, har_valid)
mlp_base_mse_RV = calc_mse_rv(mlp_base_forecasts_lnRV, mlp_base_residual_vars, actual_RV_forecast, mlp_base_valid)
har_vader_mse_RV = calc_mse_rv(har_vader_forecasts_lnRV, har_vader_residual_vars, actual_RV_forecast, har_vader_valid)
har_svm_mse_RV = calc_mse_rv(har_svm_forecasts_lnRV, har_svm_residual_vars, actual_RV_forecast, har_svm_valid)
mlp_vader_mse_RV = calc_mse_rv(mlp_vader_forecasts_lnRV, mlp_vader_residual_vars, actual_RV_forecast, mlp_vader_valid)
mlp_svm_mse_RV = calc_mse_rv(mlp_svm_forecasts_lnRV, mlp_svm_residual_vars, actual_RV_forecast, mlp_svm_valid)

print(f"  HAR-R:                        {har_mse_RV:.13f} (N={har_valid.sum()})" if not np.isnan(har_mse_RV) else f"  HAR-R:                        N/A")
print(f"  MLP:                           {mlp_base_mse_RV:.13f} (N={mlp_base_valid.sum()})" if not np.isnan(mlp_base_mse_RV) else f"  MLP:                           N/A")
print(f"  HAR-R + VADER:                 {har_vader_mse_RV:.13f} (N={har_vader_valid.sum()})" if not np.isnan(har_vader_mse_RV) else f"  HAR-R + VADER:                 N/A")
print(f"  HAR-R + SVM:                   {har_svm_mse_RV:.13f} (N={har_svm_valid.sum()})" if not np.isnan(har_svm_mse_RV) else f"  HAR-R + SVM:                   N/A")
print(f"  MLP + VADER:                   {mlp_vader_mse_RV:.13f} (N={mlp_vader_valid.sum()})" if not np.isnan(mlp_vader_mse_RV) else f"  MLP + VADER:                   N/A")
print(f"  MLP + SVM:                     {mlp_svm_mse_RV:.13f} (N={mlp_svm_valid.sum()})" if not np.isnan(mlp_svm_mse_RV) else f"  MLP + SVM:                     N/A")

# Save forecast results
forecast_results = pd.DataFrame({
    "date": forecast_dates,
    "actual_lnRV": actual_lnRV,
    "har_forecast_lnRV": har_forecasts_lnRV,
    "mlp_base_forecast_lnRV": mlp_base_forecasts_lnRV,
    "har_vader_forecast_lnRV": har_vader_forecasts_lnRV,
    "har_svm_forecast_lnRV": har_svm_forecasts_lnRV,
    "mlp_vader_forecast_lnRV": mlp_vader_forecasts_lnRV,
    "mlp_svm_forecast_lnRV": mlp_svm_forecasts_lnRV,
    "har_residual_var": har_residual_vars,
    "mlp_base_residual_var": mlp_base_residual_vars,
    "har_vader_residual_var": har_vader_residual_vars,
    "har_svm_residual_var": har_svm_residual_vars,
    "mlp_vader_residual_var": mlp_vader_residual_vars,
    "mlp_svm_residual_var": mlp_svm_residual_vars
})
forecast_output = processed_dir / "svm2_vader_combined_forecasts.csv"
forecast_results.to_csv(forecast_output, index=False)
print(f"\n✅ Forecast results saved to: {forecast_output}")

# =====================================================
# 9. BACK-TRANSFORM FOR PLOTTING ONLY
# =====================================================
print("\n" + "="*60)
print("9. BACK-TRANSFORMING FOR VISUALIZATION")
print("="*60)

# Back-transform using: RV_hat = exp(ln_RV_hat + 0.5 * sigma_hat^2)
har_forecasts_RV = np.where(
    np.isnan(har_forecasts_lnRV) | np.isnan(har_residual_vars),
    np.nan,
    np.exp(har_forecasts_lnRV + 0.5 * har_residual_vars)
)
mlp_base_forecasts_RV = np.where(
    np.isnan(mlp_base_forecasts_lnRV) | np.isnan(mlp_base_residual_vars),
    np.nan,
    np.exp(mlp_base_forecasts_lnRV + 0.5 * mlp_base_residual_vars)
)
har_vader_forecasts_RV = np.where(
    np.isnan(har_vader_forecasts_lnRV) | np.isnan(har_vader_residual_vars),
    np.nan,
    np.exp(har_vader_forecasts_lnRV + 0.5 * har_vader_residual_vars)
)
har_svm_forecasts_RV = np.where(
    np.isnan(har_svm_forecasts_lnRV) | np.isnan(har_svm_residual_vars),
    np.nan,
    np.exp(har_svm_forecasts_lnRV + 0.5 * har_svm_residual_vars)
)
mlp_vader_forecasts_RV = np.where(
    np.isnan(mlp_vader_forecasts_lnRV) | np.isnan(mlp_vader_residual_vars),
    np.nan,
    np.exp(mlp_vader_forecasts_lnRV + 0.5 * mlp_vader_residual_vars)
)
mlp_svm_forecasts_RV = np.where(
    np.isnan(mlp_svm_forecasts_lnRV) | np.isnan(mlp_svm_residual_vars),
    np.nan,
    np.exp(mlp_svm_forecasts_lnRV + 0.5 * mlp_svm_residual_vars)
)

# Get actual RV values
actual_RV = daily.iloc[start_idx:]["RV"].values

# =====================================================
# 10. TIME-SERIES PLOT
# =====================================================
print("\n" + "="*60)
print("10. CREATING TIME-SERIES PLOT")
print("="*60)

# Create date index for plotting
plot_dates = pd.to_datetime(forecast_dates)

plt.figure(figsize=(16, 10))
plt.plot(plot_dates, actual_RV, label="Actual RV", linewidth=2, color="black")

# Plot all forecasts
har_rv_valid = ~np.isnan(har_forecasts_RV) & np.isfinite(har_forecasts_RV)
mlp_base_rv_valid = ~np.isnan(mlp_base_forecasts_RV) & np.isfinite(mlp_base_forecasts_RV)
har_vader_rv_valid = ~np.isnan(har_vader_forecasts_RV) & np.isfinite(har_vader_forecasts_RV)
har_svm_rv_valid = ~np.isnan(har_svm_forecasts_RV) & np.isfinite(har_svm_forecasts_RV)
mlp_vader_rv_valid = ~np.isnan(mlp_vader_forecasts_RV) & np.isfinite(mlp_vader_forecasts_RV)
mlp_svm_rv_valid = ~np.isnan(mlp_svm_forecasts_RV) & np.isfinite(mlp_svm_forecasts_RV)

if har_rv_valid.sum() > 0:
    plt.plot(plot_dates[har_rv_valid], har_forecasts_RV[har_rv_valid], label="HAR-R", linewidth=1.5, linestyle="--", color="blue", alpha=0.7)
if mlp_base_rv_valid.sum() > 0:
    plt.plot(plot_dates[mlp_base_rv_valid], mlp_base_forecasts_RV[mlp_base_rv_valid], label="MLP", linewidth=1.5, linestyle="--", color="orange", alpha=0.7)
if har_vader_rv_valid.sum() > 0:
    plt.plot(plot_dates[har_vader_rv_valid], har_vader_forecasts_RV[har_vader_rv_valid], label="HAR-R + VADER", linewidth=1.5, linestyle="--", color="green", alpha=0.7)
if har_svm_rv_valid.sum() > 0:
    plt.plot(plot_dates[har_svm_rv_valid], har_svm_forecasts_RV[har_svm_rv_valid], label="HAR-R + SVM", linewidth=1.5, linestyle="--", color="red", alpha=0.7)
if mlp_vader_rv_valid.sum() > 0:
    plt.plot(plot_dates[mlp_vader_rv_valid], mlp_vader_forecasts_RV[mlp_vader_rv_valid], label="MLP + VADER", linewidth=1.5, linestyle="--", color="purple", alpha=0.7)
if mlp_svm_rv_valid.sum() > 0:
    plt.plot(plot_dates[mlp_svm_rv_valid], mlp_svm_forecasts_RV[mlp_svm_rv_valid], label="MLP + SVM", linewidth=1.5, linestyle="--", color="brown", alpha=0.7)

plt.xlabel("Date", fontsize=12)
plt.ylabel("Realized Variance (RV)", fontsize=12)
plt.title("Realized Variance Forecasts: HAR-R vs MLP vs HAR-R+VADER vs HAR-R+SVM vs MLP+VADER vs MLP+SVM", fontsize=14, fontweight="bold")
plt.legend(loc="upper left", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_output = processed_dir / "svm2_vader_combined_forecast_plot.png"
plt.savefig(plot_output, dpi=300, bbox_inches="tight")
print(f"✅ Plot saved to: {plot_output}")
plt.close()

# =====================================================
# 11. SAVE COMPREHENSIVE ANALYSIS RESULTS
# =====================================================
svm2_output_dir = BASE_DIR / "data" / "processed"
svm2_output_dir.mkdir(parents=True, exist_ok=True)

# Model comparison summary
model_comparison = pd.DataFrame({
    "Model": ["HAR-R", "MLP", "HAR-R + VADER", "HAR-R + SVM", "MLP + VADER", "MLP + SVM"],
    "In_Sample_R2": [
        har_diagnostics["R²"],
        mlp_base_diagnostics["R²"],
        har_vader_diagnostics["R²"],
        har_svm_diagnostics["R²"],
        mlp_vader_diagnostics["R²"],
        mlp_svm_diagnostics["R²"]
    ],
    "In_Sample_Adj_R2": [
        har_diagnostics["Adj R²"],
        mlp_base_diagnostics["Adj R²"],
        har_vader_diagnostics["Adj R²"],
        har_svm_diagnostics["Adj R²"],
        mlp_vader_diagnostics["Adj R²"],
        mlp_svm_diagnostics["Adj R²"]
    ],
    "In_Sample_AIC": [
        har_diagnostics["AIC"],
        mlp_base_diagnostics["AIC"],
        har_vader_diagnostics["AIC"],
        har_svm_diagnostics["AIC"],
        mlp_vader_diagnostics["AIC"],
        mlp_svm_diagnostics["AIC"]
    ],
    "In_Sample_BIC": [
        har_diagnostics["BIC"],
        mlp_base_diagnostics["BIC"],
        har_vader_diagnostics["BIC"],
        har_svm_diagnostics["BIC"],
        mlp_vader_diagnostics["BIC"],
        mlp_svm_diagnostics["BIC"]
    ],
    "Out_Sample_MSE_lnRV": [
        har_mse_lnRV if not np.isnan(har_mse_lnRV) else np.nan,
        mlp_base_mse_lnRV if not np.isnan(mlp_base_mse_lnRV) else np.nan,
        har_vader_mse_lnRV if not np.isnan(har_vader_mse_lnRV) else np.nan,
        har_svm_mse_lnRV if not np.isnan(har_svm_mse_lnRV) else np.nan,
        mlp_vader_mse_lnRV if not np.isnan(mlp_vader_mse_lnRV) else np.nan,
        mlp_svm_mse_lnRV if not np.isnan(mlp_svm_mse_lnRV) else np.nan
    ],
    "Out_Sample_MSE_RV": [
        har_mse_RV if not np.isnan(har_mse_RV) else np.nan,
        mlp_base_mse_RV if not np.isnan(mlp_base_mse_RV) else np.nan,
        har_vader_mse_RV if not np.isnan(har_vader_mse_RV) else np.nan,
        har_svm_mse_RV if not np.isnan(har_svm_mse_RV) else np.nan,
        mlp_vader_mse_RV if not np.isnan(mlp_vader_mse_RV) else np.nan,
        mlp_svm_mse_RV if not np.isnan(mlp_svm_mse_RV) else np.nan
    ],
    "Valid_Forecasts": [
        har_valid.sum(),
        mlp_base_valid.sum(),
        har_vader_valid.sum(),
        har_svm_valid.sum(),
        mlp_vader_valid.sum(),
        mlp_svm_valid.sum()
    ]
})

# Save model comparison to CSV
comparison_output = svm2_output_dir / "svm2_vader_combined_comparison_results.csv"
model_comparison.to_csv(comparison_output, index=False)
print(f"✅ Model comparison saved to: {comparison_output}")

# Format MSE values for display (handle NaN values)
har_mse_lnRV_str = f"{har_mse_lnRV:.8f}" if not np.isnan(har_mse_lnRV) else "N/A"
mlp_base_mse_lnRV_str = f"{mlp_base_mse_lnRV:.8f}" if not np.isnan(mlp_base_mse_lnRV) else "N/A"
har_vader_mse_lnRV_str = f"{har_vader_mse_lnRV:.8f}" if not np.isnan(har_vader_mse_lnRV) else "N/A"
har_svm_mse_lnRV_str = f"{har_svm_mse_lnRV:.8f}" if not np.isnan(har_svm_mse_lnRV) else "N/A"
mlp_vader_mse_lnRV_str = f"{mlp_vader_mse_lnRV:.8f}" if not np.isnan(mlp_vader_mse_lnRV) else "N/A"
mlp_svm_mse_lnRV_str = f"{mlp_svm_mse_lnRV:.8f}" if not np.isnan(mlp_svm_mse_lnRV) else "N/A"

har_mse_RV_str = f"{har_mse_RV:.13f}" if not np.isnan(har_mse_RV) else "N/A"
mlp_base_mse_RV_str = f"{mlp_base_mse_RV:.13f}" if not np.isnan(mlp_base_mse_RV) else "N/A"
har_vader_mse_RV_str = f"{har_vader_mse_RV:.13f}" if not np.isnan(har_vader_mse_RV) else "N/A"
har_svm_mse_RV_str = f"{har_svm_mse_RV:.13f}" if not np.isnan(har_svm_mse_RV) else "N/A"
mlp_vader_mse_RV_str = f"{mlp_vader_mse_RV:.13f}" if not np.isnan(mlp_vader_mse_RV) else "N/A"
mlp_svm_mse_RV_str = f"{mlp_svm_mse_RV:.13f}" if not np.isnan(mlp_svm_mse_RV) else "N/A"

# Create detailed summary text file
summary_text = f"""
================================================================================
SVM2 + VADER COMBINED SENTIMENT-BASED VOLATILITY FORECASTING - ANALYSIS RESULTS
================================================================================

Analysis Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
Data Period: {daily['date'].min().strftime('%Y-%m-%d')} to {daily['date'].max().strftime('%Y-%m-%d')}
Total Days: {len(daily):,}
Training Window: {INITIAL_WINDOW_SIZE} days
Forecast Days: {len(daily) - start_idx:,}

================================================================================
IN-SAMPLE MODEL PERFORMANCE
================================================================================

HAR-R:
  R²: {har_diagnostics['R²']:.6f}
  Adj R²: {har_diagnostics['Adj R²']:.6f}
  AIC: {har_diagnostics['AIC']:.6f}
  BIC: {har_diagnostics['BIC']:.6f}

MLP:
  R²: {mlp_base_diagnostics['R²']:.6f}
  Adj R²: {mlp_base_diagnostics['Adj R²']:.6f}
  AIC: {mlp_base_diagnostics['AIC']:.6f}
  BIC: {mlp_base_diagnostics['BIC']:.6f}

HAR-R + VADER:
  R²: {har_vader_diagnostics['R²']:.6f}
  Adj R²: {har_vader_diagnostics['Adj R²']:.6f}
  AIC: {har_vader_diagnostics['AIC']:.6f}
  BIC: {har_vader_diagnostics['BIC']:.6f}

HAR-R + SVM:
  R²: {har_svm_diagnostics['R²']:.6f}
  Adj R²: {har_svm_diagnostics['Adj R²']:.6f}
  AIC: {har_svm_diagnostics['AIC']:.6f}
  BIC: {har_svm_diagnostics['BIC']:.6f}

MLP + VADER:
  R²: {mlp_vader_diagnostics['R²']:.6f}
  Adj R²: {mlp_vader_diagnostics['Adj R²']:.6f}
  AIC: {mlp_vader_diagnostics['AIC']:.6f}
  BIC: {mlp_vader_diagnostics['BIC']:.6f}

MLP + SVM:
  R²: {mlp_svm_diagnostics['R²']:.6f}
  Adj R²: {mlp_svm_diagnostics['Adj R²']:.6f}
  AIC: {mlp_svm_diagnostics['AIC']:.6f}
  BIC: {mlp_svm_diagnostics['BIC']:.6f}

================================================================================
OUT-OF-SAMPLE FORECAST PERFORMANCE (ln(RV) space)
================================================================================

HAR-R:                        {har_mse_lnRV_str} (N={har_valid.sum()})
MLP:                           {mlp_base_mse_lnRV_str} (N={mlp_base_valid.sum()})
HAR-R + VADER:                 {har_vader_mse_lnRV_str} (N={har_vader_valid.sum()})
HAR-R + SVM:                   {har_svm_mse_lnRV_str} (N={har_svm_valid.sum()})
MLP + VADER:                   {mlp_vader_mse_lnRV_str} (N={mlp_vader_valid.sum()})
MLP + SVM:                     {mlp_svm_mse_lnRV_str} (N={mlp_svm_valid.sum()})

================================================================================
OUT-OF-SAMPLE FORECAST PERFORMANCE (RV space, back-transformed)
================================================================================

HAR-R:                        {har_mse_RV_str} (N={har_valid.sum()})
MLP:                           {mlp_base_mse_RV_str} (N={mlp_base_valid.sum()})
HAR-R + VADER:                 {har_vader_mse_RV_str} (N={har_vader_valid.sum()})
HAR-R + SVM:                   {har_svm_mse_RV_str} (N={har_svm_valid.sum()})
MLP + VADER:                   {mlp_vader_mse_RV_str} (N={mlp_vader_valid.sum()})
MLP + SVM:                     {mlp_svm_mse_RV_str} (N={mlp_svm_valid.sum()})

================================================================================
MODEL RANKINGS (by Out-of-Sample MSE in ln(RV) space)
================================================================================
"""

# Rank models by MSE (lower is better)
mse_ranking = []
if not np.isnan(har_mse_lnRV):
    mse_ranking.append(("HAR-R", har_mse_lnRV))
if not np.isnan(mlp_base_mse_lnRV):
    mse_ranking.append(("MLP", mlp_base_mse_lnRV))
if not np.isnan(har_vader_mse_lnRV):
    mse_ranking.append(("HAR-R + VADER", har_vader_mse_lnRV))
if not np.isnan(har_svm_mse_lnRV):
    mse_ranking.append(("HAR-R + SVM", har_svm_mse_lnRV))
if not np.isnan(mlp_vader_mse_lnRV):
    mse_ranking.append(("MLP + VADER", mlp_vader_mse_lnRV))
if not np.isnan(mlp_svm_mse_lnRV):
    mse_ranking.append(("MLP + SVM", mlp_svm_mse_lnRV))

mse_ranking.sort(key=lambda x: x[1])
for rank, (model, mse) in enumerate(mse_ranking, 1):
    summary_text += f"{rank}. {model}: {mse:.8f}\n"

summary_text += f"""
================================================================================
FILES GENERATED
================================================================================

1. Model Diagnostics: {diagnostics_output}
2. Forecast Results: {forecast_output}
3. Model Comparison: {comparison_output}
4. Visualization: {plot_output}

================================================================================
NOTES
================================================================================

- MSE evaluation is in ln(RV) space (no back-transformation)
- Back-transformation used only for visualization
- Expanding window approach: training set grows as we forecast forward
- Initial training window: {INITIAL_WINDOW_SIZE} days (~6 months)
- All HAR models use HAC standard errors (robust methods)
- Comparison includes: HAR-R vs MLP vs HAR-R+VADER vs HAR-R+SVM vs MLP+VADER vs MLP+SVM

================================================================================
"""

# Save summary text file
summary_output = svm2_output_dir / "svm2_vader_combined_analysis_summary.txt"
with open(summary_output, 'w') as f:
    f.write(summary_text)
print(f"✅ Analysis summary saved to: {summary_output}")

# =====================================================
# 12. EXPORT HAR-R COEFFICIENT TABLE (LaTeX)
# =====================================================
print("\n" + "="*60)
print("12. EXPORTING HAR-R COEFFICIENT TABLE (LaTeX)")
print("="*60)

def format_coefficient(value, use_scientific=False):
    """Format coefficient with appropriate precision and scientific notation if needed."""
    if abs(value) < 0.0001 and value != 0:
        return f"{value:.4e}"
    elif abs(value) < 0.01:
        return f"{value:.4f}"
    else:
        return f"{value:.4f}"

def format_std_error(value):
    """Format standard error (without parentheses - added in LaTeX)."""
    if abs(value) < 0.0001 and value != 0:
        return f"{value:.4e}"
    elif abs(value) < 0.01:
        return f"{value:.4f}"
    else:
        return f"{value:.3f}"

def get_significance_stars(pvalue):
    """Return significance stars based on p-value."""
    if pvalue < 0.01:
        return "***"
    elif pvalue < 0.05:
        return "**"
    elif pvalue < 0.10:
        return "*"
    else:
        return ""

# Extract coefficients, standard errors, and p-values
har_coefs = har_model.params
har_stderr = har_model.bse
har_pvalues = har_model.pvalues

har_vader_coefs = har_vader_model.params
har_vader_stderr = har_vader_model.bse
har_vader_pvalues = har_vader_model.pvalues

har_svm_coefs = har_svm_model.params
har_svm_stderr = har_svm_model.bse
har_svm_pvalues = har_svm_model.pvalues

# Get variable names
har_var_names = har_model.params.index.tolist()
har_vader_var_names = har_vader_model.params.index.tolist()
har_svm_var_names = har_svm_model.params.index.tolist()

# Create LaTeX table
latex_lines = [
    "\\begin{table}[t]",
    "\\centering",
    "\\caption{HAR-R Models With and Without Sentiment Indices (2021)}",
    "\\label{tab:har_sentiment}",
    "\\small",
    "\\renewcommand{\\arraystretch}{0.95}",
    "\\begin{tabular}{lccc}",
    "\\hline",
    " & Baseline HAR-R & HAR-R + VADER & HAR-R + SVM \\\\",
    "\\hline"
]

# Map variable names to LaTeX format
var_latex_map = {
    "const": "Constant",
    "ln(RV_daily_t-1)": "$\\ln RV_{t-1}^{(d)}$",
    "ln(RV_weekly_t-1)": "$\\ln RV_{t-1}^{(w)}$",
    "ln(RV_monthly_t-1)": "$\\ln RV_{t-1}^{(m)}$",
    "vader_sentiment_t-1": "$\\text{VADER}_{t-1}$",
    "vader_disagreement_t-1": "$\\text{VADER Disagreement}_{t-1}$",
    "svm_sentiment_t-1": "$\\text{SVM}_{t-1}$",
    "svm_disagreement_t-1": "$\\text{SVM Disagreement}_{t-1}$"
}

# Add coefficient rows
all_vars = set(har_var_names) | set(har_vader_var_names) | set(har_svm_var_names)
# Order: constant, lnRV lags, then sentiment variables
ordered_vars = []
if "const" in all_vars:
    ordered_vars.append("const")
for var in ["ln(RV_daily_t-1)", "ln(RV_weekly_t-1)", "ln(RV_monthly_t-1)"]:
    if var in all_vars:
        ordered_vars.append(var)
for var in all_vars:
    if var not in ordered_vars:
        ordered_vars.append(var)

for var in ordered_vars:
    var_display = var_latex_map.get(var, var.replace("_", "\\_"))
    
    # Get coefficient and std error for each model
    har_coef = har_coefs[var] if var in har_var_names else np.nan
    har_se = har_stderr[var] if var in har_var_names else np.nan
    har_pval = har_pvalues[var] if var in har_var_names else np.nan
    
    har_vader_coef = har_vader_coefs[var] if var in har_vader_var_names else np.nan
    har_vader_se = har_vader_stderr[var] if var in har_vader_var_names else np.nan
    har_vader_pval = har_vader_pvalues[var] if var in har_vader_var_names else np.nan
    
    har_svm_coef = har_svm_coefs[var] if var in har_svm_var_names else np.nan
    har_svm_se = har_svm_stderr[var] if var in har_svm_var_names else np.nan
    har_svm_pval = har_svm_pvalues[var] if var in har_svm_var_names else np.nan
    
    # Format coefficient row (coefficient with stars)
    coef_row = [var_display]
    
    # Baseline HAR-R
    if not np.isnan(har_coef):
        stars = get_significance_stars(har_pval)
        coef_str = format_coefficient(har_coef)
        coef_row.append(f" & {coef_str}{stars}")
    else:
        coef_row.append(" & ")
    
    # HAR-R + VADER
    if not np.isnan(har_vader_coef):
        stars = get_significance_stars(har_vader_pval)
        coef_str = format_coefficient(har_vader_coef)
        coef_row.append(f" & {coef_str}{stars}")
    else:
        coef_row.append(" & ")
    
    # HAR-R + SVM
    if not np.isnan(har_svm_coef):
        stars = get_significance_stars(har_svm_pval)
        coef_str = format_coefficient(har_svm_coef)
        coef_row.append(f" & {coef_str}{stars}")
    else:
        coef_row.append(" & ")
    
    latex_lines.append("".join(coef_row) + " \\\\")
    
    # Add standard error row (in parentheses)
    se_row = [" & "]
    if not np.isnan(har_coef):
        se_str = format_std_error(har_se)
        se_row.append(f"({se_str})")
    else:
        se_row.append("")
    
    if not np.isnan(har_vader_coef):
        se_str = format_std_error(har_vader_se)
        se_row.append(f" & ({se_str})")
    else:
        se_row.append(" & ")
    
    if not np.isnan(har_svm_coef):
        se_str = format_std_error(har_svm_se)
        se_row.append(f" & ({se_str})")
    else:
        se_row.append(" & ")
    
    latex_lines.append("".join(se_row) + " \\\\")
    latex_lines.append("")  # Empty line for spacing

# Add model diagnostics
latex_lines.extend([
    "\\hline",
    "Observations",
    f" & {har_diagnostics['N']}",
    f" & {har_vader_diagnostics['N']}",
    f" & {har_svm_diagnostics['N']} \\\\",
    "",
    "$R^2$",
    f" & {har_diagnostics['R²']:.4f}",
    f" & {har_vader_diagnostics['R²']:.4f}",
    f" & {har_svm_diagnostics['R²']:.4f} \\\\",
    "",
    "Adjusted $R^2$",
    f" & {har_diagnostics['Adj R²']:.4f}",
    f" & {har_vader_diagnostics['Adj R²']:.4f}",
    f" & {har_svm_diagnostics['Adj R²']:.4f} \\\\",
    "",
    "AIC",
    f" & {har_diagnostics['AIC']:.2f}",
    f" & {har_vader_diagnostics['AIC']:.2f}",
    f" & {har_svm_diagnostics['AIC']:.2f} \\\\",
    "",
    "BIC",
    f" & {har_diagnostics['BIC']:.2f}",
    f" & {har_vader_diagnostics['BIC']:.2f}",
    f" & {har_svm_diagnostics['BIC']:.2f} \\\\",
    "\\hline",
    "\\end{tabular}",
    "",
    "\\begin{minipage}{0.9\\textwidth}",
    "\\footnotesize",
    "\\textit{Notes:} HAC standard errors are reported in parentheses.",
    "*** $p < 0.01$, ** $p < 0.05$, * $p < 0.10$.",
    f"The sample covers calendar year 2021, with the first {har_diagnostics['N']} observations used for in-sample estimation.",
    "\\end{minipage}",
    "\\end{table}"
])

latex_table = "\n".join(latex_lines)

# Save LaTeX table
latex_output = svm2_output_dir / "har_coefficient_table.tex"
with open(latex_output, 'w') as f:
    f.write(latex_table)
print(f"✅ LaTeX coefficient table saved to: {latex_output}")

# Also print a preview
print("\n" + "="*60)
print("LaTeX Table Preview (first 30 lines):")
print("="*60)
print("\n".join(latex_lines[:30]))
print("\n... (table continues)")
print("="*60)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nSummary:")
print(f"  - In-sample diagnostics: {diagnostics_output}")
print(f"  - Out-of-sample forecasts: {forecast_output}")
print(f"  - Model comparison: {comparison_output}")
print(f"  - Analysis summary: {summary_output}")
print(f"  - Visualization: {plot_output}")
print(f"  - LaTeX coefficient table: {latex_output}")
print(f"\nNote: MSE evaluation is in ln(RV) space (no back-transformation).")
print(f"      Back-transformation used only for visualization.")
print(f"\nModels compared:")
print(f"  1. HAR-R (baseline)")
print(f"  2. MLP (baseline)")
print(f"  3. HAR-R + VADER")
print(f"  4. HAR-R + SVM")
print(f"  5. MLP + VADER")
print(f"  6. MLP + SVM")

