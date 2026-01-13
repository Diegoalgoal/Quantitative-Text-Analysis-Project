# =====================================================
# VADER SENTIMENT-BASED VOLATILITY FORECASTING
# HAR, HAR+VADER, MLP+VADER Models
# =====================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
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

# Input files
REDDIT_CLEAN_FILE = BASE_DIR / "data" / "raw" / "Reddit_2021.csv"
BTC_FILE = BASE_DIR / "data" / "raw" / "btc_5min_2021.csv"

# Initial training window size (6 months ≈ 180 days)
# Will use expanding window: start with INITIAL_WINDOW_SIZE, then expand
INITIAL_WINDOW_SIZE = 180

# =====================================================
# 1. LOAD AND PROCESS REDDIT DATA
# =====================================================
print("="*60)
print("1. LOADING AND PROCESSING REDDIT DATA")
print("="*60)
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
reddit = reddit.dropna(subset=["hour"])
reddit["hour"] = reddit["hour"].dt.floor("h")

# Filter to 2021 only
reddit = reddit[reddit["hour"].dt.year == 2021].copy()
print(f"Reddit data filtered to 2021: {len(reddit):,} observations")

# =====================================================
# 2. APPLY VADER SENTIMENT
# =====================================================
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

# =====================================================
# 3. LOAD BTC DATA (5-MINUTE)
# =====================================================
print("\n" + "="*60)
print("2. LOADING BTC DATA (5-MINUTE)")
print("="*60)
if not BTC_FILE.exists():
    raise FileNotFoundError(f"BTC 5-minute data file not found: {BTC_FILE}\n"
                           f"Please run: python3 Second_Run/fetch_btc_5min_eodhd.py")

print("Loading BTC 5-minute data...")
btc = pd.read_csv(BTC_FILE, parse_dates=["datetime"])
btc = btc.sort_values("datetime")

# Ensure datetime is timezone-aware (UTC)
if btc["datetime"].dt.tz is None:
    btc["datetime"] = btc["datetime"].dt.tz_localize("UTC")
else:
    btc["datetime"] = btc["datetime"].dt.tz_convert("UTC")

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
print("3. CONSTRUCTING DAILY REALIZED VARIANCE")
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

# =====================================================
# 5. MERGE BTC RV + VADER (HOURLY TO DAILY)
# =====================================================
print("Merging BTC daily RV with hourly VADER data...")
# Aggregate VADER to daily (take mean of hourly sentiment/disagreement)
daily_vader = (
    hourly_vader.groupby(hourly_vader["hour"].dt.date)
    .agg(
        sentiment=("vader_sentiment_mean", "mean"),
        disagreement=("vader_sentiment_std", "mean"),
    )
    .reset_index()
)
daily_vader.rename(columns={"hour": "date"}, inplace=True)

# Merge daily RV with daily VADER
daily = pd.merge(daily_rv, daily_vader, on="date", how="inner")
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

# Lag sentiment and disagreement by one day
daily["sentiment_lag1"] = daily["sentiment"].shift(1)
daily["disagreement_lag1"] = daily["disagreement"].shift(1)

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
print("4. IN-SAMPLE MODEL ESTIMATION")
print("="*60)

# Use initial training window for in-sample diagnostics
# This ensures consistency with out-of-sample evaluation
train_end_idx = INITIAL_WINDOW_SIZE
if train_end_idx >= len(daily):
    raise ValueError(f"Initial window size ({INITIAL_WINDOW_SIZE}) exceeds available data ({len(daily)} days)")

train_data = daily.iloc[:train_end_idx].copy()
print(f"Using first {len(train_data)} days for in-sample estimation")
print(f"Training period: {train_data['date'].min()} to {train_data['date'].max()}")

# Prepare data for in-sample estimation
# Use DataFrames with meaningful names for OLS models
X_base_df = train_data[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"]].copy()
X_base_df.columns = ["ln(RV_daily_t-1)", "ln(RV_weekly_t-1)", "ln(RV_monthly_t-1)"]
X_base_const = sm.add_constant(X_base_df)

X_sent_df = train_data[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30", "sentiment_lag1", "disagreement_lag1"]].copy()
X_sent_df.columns = ["ln(RV_daily_t-1)", "ln(RV_weekly_t-1)", "ln(RV_monthly_t-1)", "sentiment_t-1", "disagreement_t-1"]
X_sent_const = sm.add_constant(X_sent_df)

# For MLP, use numpy arrays (scaling)
X_base = train_data[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"]].values
X_sent = train_data[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30", "sentiment_lag1", "disagreement_lag1"]].values
y = train_data["lnRV"].values

# Scale features for MLP
scaler = StandardScaler()
X_base_scaled = scaler.fit_transform(X_base)
X_sent_scaled = scaler.fit_transform(X_sent)

# Model 1: HAR-Robust (Baseline with HAC standard errors)
print("\n--- HAR-Robust (Baseline) ---")
har_model = sm.OLS(y, X_base_const).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
har_diagnostics = {
    "Model": "HAR-Robust",
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
print(f"F-statistic: {har_diagnostics['F-statistic']:.6f} (p-value: {har_diagnostics['F p-value']:.6e})")
print(f"Log-likelihood: {har_diagnostics['Log-likelihood']:.6f}")
print(f"AIC: {har_diagnostics['AIC']:.6f}")
print(f"BIC: {har_diagnostics['BIC']:.6f}")
print("\nCoefficient Statistics:")
print(har_model.summary().tables[1])

# Model 2: HAR-Robust + VADER
print("\n--- HAR-Robust + VADER ---")
har_sent_model = sm.OLS(y, X_sent_const).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
har_sent_diagnostics = {
    "Model": "HAR-Robust + VADER",
    "N": har_sent_model.nobs,
    "R²": har_sent_model.rsquared,
    "Adj R²": har_sent_model.rsquared_adj,
    "F-statistic": har_sent_model.fvalue,
    "F p-value": har_sent_model.f_pvalue,
    "Log-likelihood": har_sent_model.llf,
    "AIC": har_sent_model.aic,
    "BIC": har_sent_model.bic,
    "Residual variance": har_sent_model.mse_resid
}
print(f"N: {har_sent_diagnostics['N']}")
print(f"R²: {har_sent_diagnostics['R²']:.6f}")
print(f"Adj R²: {har_sent_diagnostics['Adj R²']:.6f}")
print(f"F-statistic: {har_sent_diagnostics['F-statistic']:.6f} (p-value: {har_sent_diagnostics['F p-value']:.6e})")
print(f"Log-likelihood: {har_sent_diagnostics['Log-likelihood']:.6f}")
print(f"AIC: {har_sent_diagnostics['AIC']:.6f}")
print(f"BIC: {har_sent_diagnostics['BIC']:.6f}")
print("\nCoefficient Statistics:")
print(har_sent_model.summary().tables[1])

# Model 3: MLP (Baseline)
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

# Calculate diagnostics for MLP baseline
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
print(f"F-statistic: N/A (non-linear model)")
print(f"Log-likelihood: {mlp_base_diagnostics['Log-likelihood']:.6f}")
print(f"AIC: {mlp_base_diagnostics['AIC']:.6f}")
print(f"BIC: {mlp_base_diagnostics['BIC']:.6f}")

# Model 4: MLP + VADER
print("\n--- MLP + VADER ---")
mlp_sent_model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation='tanh',
    solver='adam',
    max_iter=2000,  # Increased to reduce convergence warnings
    tol=1e-4,
    random_state=42,
    early_stopping=False  # Disable early stopping for consistency
)
mlp_sent_model.fit(X_sent_scaled, y)
mlp_sent_pred = mlp_sent_model.predict(X_sent_scaled)

# Calculate diagnostics for MLP (manual calculation)
n = len(y)
ss_res = np.sum((y - mlp_sent_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)
r2_adj = 1 - (1 - r2) * (n - 1) / (n - X_sent.shape[1] - 1)
mse = ss_res / n
log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(mse) + 1)
aic = -2 * log_likelihood + 2 * (X_sent.shape[1] + 1)  # Approximate
bic = -2 * log_likelihood + np.log(n) * (X_sent.shape[1] + 1)  # Approximate

mlp_sent_diagnostics = {
    "Model": "MLP + VADER",
    "N": n,
    "R²": r2,
    "Adj R²": r2_adj,
    "F-statistic": np.nan,  # Not directly available for MLP
    "F p-value": np.nan,
    "Log-likelihood": log_likelihood,
    "AIC": aic,
    "BIC": bic,
    "Residual variance": mse
}
print(f"N: {mlp_sent_diagnostics['N']}")
print(f"R²: {mlp_sent_diagnostics['R²']:.6f}")
print(f"Adj R²: {mlp_sent_diagnostics['Adj R²']:.6f}")
print(f"F-statistic: N/A (non-linear model)")
print(f"Log-likelihood: {mlp_sent_diagnostics['Log-likelihood']:.6f}")
print(f"AIC: {mlp_sent_diagnostics['AIC']:.6f}")
print(f"BIC: {mlp_sent_diagnostics['BIC']:.6f}")

# Save diagnostics to DataFrame
diagnostics_df = pd.DataFrame([
    har_diagnostics,
    har_sent_diagnostics,
    mlp_base_diagnostics,
    mlp_sent_diagnostics
])
diagnostics_output = processed_dir / "vader_model_diagnostics.csv"
diagnostics_df.to_csv(diagnostics_output, index=False)
print(f"\n✅ In-sample diagnostics saved to: {diagnostics_output}")

# =====================================================
# 7. EXPANDING WINDOW OUT-OF-SAMPLE FORECASTING
# =====================================================
print("\n" + "="*60)
print("5. EXPANDING WINDOW OUT-OF-SAMPLE FORECASTING")
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
har_sent_forecasts_lnRV = []
mlp_base_forecasts_lnRV = []
mlp_sent_forecasts_lnRV = []
actual_lnRV = []
forecast_dates = []

# Storage for residual variances (for back-transformation)
har_residual_vars = []
har_sent_residual_vars = []
mlp_base_residual_vars = []
mlp_sent_residual_vars = []

print("\nGenerating expanding window forecasts...")
for i in range(start_idx, len(daily)):
    if (i - start_idx) % 50 == 0:
        print(f"  Processing day {i - start_idx + 1}/{len(daily) - start_idx}...")
    
    # Expanding window: use all data from start up to (but not including) current day
    # This ensures same window choice across all models
    train_window = daily.iloc[:i].copy()
    test_row = daily.iloc[i:i+1].copy()
    
    # Prepare training data
    X_train_base = train_window[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"]].values
    X_train_sent = train_window[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30", 
                                  "sentiment_lag1", "disagreement_lag1"]].values
    y_train = train_window["lnRV"].values
    
    # Prepare test data
    X_test_base = test_row[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"]].values
    X_test_sent = test_row[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30", 
                             "sentiment_lag1", "disagreement_lag1"]].values
    y_test = test_row["lnRV"].values[0]
    
    # Ensure test data is 2D (single row)
    if X_test_base.ndim == 1:
        X_test_base = X_test_base.reshape(1, -1)
    if X_test_sent.ndim == 1:
        X_test_sent = X_test_sent.reshape(1, -1)
    
    # Scale features for MLP (separate scalers for base and sentiment models)
    scaler_base = StandardScaler()
    scaler_sent = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    X_train_sent_scaled = scaler_sent.fit_transform(X_train_sent)
    X_test_base_scaled = scaler_base.transform(X_test_base)
    X_test_sent_scaled = scaler_sent.transform(X_test_sent)
    
    # Add constant for OLS (ensure 2D arrays)
    X_train_base_const = sm.add_constant(X_train_base, has_constant='add')
    X_test_base_const = sm.add_constant(X_test_base, has_constant='add')
    X_train_sent_const = sm.add_constant(X_train_sent, has_constant='add')
    X_test_sent_const = sm.add_constant(X_test_sent, has_constant='add')
    
    # HAR-Robust forecast
    try:
        har_model_roll = sm.OLS(y_train, X_train_base_const).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
        har_pred_lnRV = har_model_roll.predict(X_test_base_const)[0]
        har_residual_var = har_model_roll.mse_resid
    except Exception as e:
        if (i - start_idx) < 5:  # Only print first few errors to avoid spam
            print(f"    Warning: HAR-Robust forecast failed at day {i-start_idx+1}: {e}")
        har_pred_lnRV = np.nan
        har_residual_var = np.nan
    
    # HAR-Robust + VADER forecast
    try:
        har_sent_model_roll = sm.OLS(y_train, X_train_sent_const).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
        har_sent_pred_lnRV = har_sent_model_roll.predict(X_test_sent_const)[0]
        har_sent_residual_var = har_sent_model_roll.mse_resid
    except Exception as e:
        if (i - start_idx) < 5:  # Only print first few errors to avoid spam
            print(f"    Warning: HAR-Robust+VADER forecast failed at day {i-start_idx+1}: {e}")
        har_sent_pred_lnRV = np.nan
        har_sent_residual_var = np.nan
    
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
        # Approximate residual variance from training residuals
        mlp_base_train_pred = mlp_base_model_roll.predict(X_train_base_scaled)
        mlp_base_residual_var = np.mean((y_train - mlp_base_train_pred) ** 2)
    except Exception as e:
        if (i - start_idx) < 5:
            print(f"    Warning: MLP baseline forecast failed at day {i-start_idx+1}: {e}")
        mlp_base_pred_lnRV = np.nan
        mlp_base_residual_var = np.nan
    
    # MLP + VADER forecast
    try:
        mlp_sent_model_roll = MLPRegressor(
            hidden_layer_sizes=(10,),
            activation='tanh',
            solver='adam',
            max_iter=2000,  # Increased to reduce convergence warnings
            tol=1e-4,
            random_state=42,
            early_stopping=False  # Disable early stopping for consistency
        )
        mlp_sent_model_roll.fit(X_train_sent_scaled, y_train)
        mlp_sent_pred_lnRV = mlp_sent_model_roll.predict(X_test_sent_scaled.reshape(1, -1))[0]
        # Approximate residual variance from training residuals
        mlp_train_pred = mlp_sent_model_roll.predict(X_train_sent_scaled)
        mlp_sent_residual_var = np.mean((y_train - mlp_train_pred) ** 2)
    except Exception as e:
        if (i - start_idx) < 5:  # Only print first few errors to avoid spam
            print(f"    Warning: MLP+VADER forecast failed at day {i-start_idx+1}: {e}")
        mlp_sent_pred_lnRV = np.nan
        mlp_sent_residual_var = np.nan
    
    # Store forecasts (in ln(RV) space)
    har_forecasts_lnRV.append(har_pred_lnRV)
    har_sent_forecasts_lnRV.append(har_sent_pred_lnRV)
    mlp_base_forecasts_lnRV.append(mlp_base_pred_lnRV)
    mlp_sent_forecasts_lnRV.append(mlp_sent_pred_lnRV)
    actual_lnRV.append(y_test)
    forecast_dates.append(test_row["date"].values[0])
    
    # Store residual variances
    har_residual_vars.append(har_residual_var)
    har_sent_residual_vars.append(har_sent_residual_var)
    mlp_base_residual_vars.append(mlp_base_residual_var)
    mlp_sent_residual_vars.append(mlp_sent_residual_var)

# =====================================================
# 8. OUT-OF-SAMPLE MSE EVALUATION (ln(RV) SPACE)
# =====================================================
print("\n" + "="*60)
print("6. OUT-OF-SAMPLE FORECAST EVALUATION")
print("="*60)

# Convert to arrays
har_forecasts_lnRV = np.array(har_forecasts_lnRV)
har_sent_forecasts_lnRV = np.array(har_sent_forecasts_lnRV)
mlp_base_forecasts_lnRV = np.array(mlp_base_forecasts_lnRV)
mlp_sent_forecasts_lnRV = np.array(mlp_sent_forecasts_lnRV)
actual_lnRV = np.array(actual_lnRV)
har_residual_vars = np.array(har_residual_vars)
har_sent_residual_vars = np.array(har_sent_residual_vars)
mlp_base_residual_vars = np.array(mlp_base_residual_vars)
mlp_sent_residual_vars = np.array(mlp_sent_residual_vars)

# Filter out NaN predictions
har_valid = ~np.isnan(har_forecasts_lnRV)
har_sent_valid = ~np.isnan(har_sent_forecasts_lnRV)
mlp_base_valid = ~np.isnan(mlp_base_forecasts_lnRV)
mlp_sent_valid = ~np.isnan(mlp_sent_forecasts_lnRV)

# Calculate MSE in ln(RV) space (NO back-transformation)
# Check if there are valid predictions before calculating MSE
if har_valid.sum() > 0:
    har_mse_lnRV = mean_squared_error(actual_lnRV[har_valid], har_forecasts_lnRV[har_valid])
else:
    har_mse_lnRV = np.nan
    print("Warning: All HAR forecasts are NaN")

if har_sent_valid.sum() > 0:
    har_sent_mse_lnRV = mean_squared_error(actual_lnRV[har_sent_valid], har_sent_forecasts_lnRV[har_sent_valid])
else:
    har_sent_mse_lnRV = np.nan
    print("Warning: All HAR + VADER forecasts are NaN")

if mlp_base_valid.sum() > 0:
    mlp_base_mse_lnRV = mean_squared_error(actual_lnRV[mlp_base_valid], mlp_base_forecasts_lnRV[mlp_base_valid])
else:
    mlp_base_mse_lnRV = np.nan
    print("Warning: All MLP baseline forecasts are NaN")

if mlp_sent_valid.sum() > 0:
    mlp_sent_mse_lnRV = mean_squared_error(actual_lnRV[mlp_sent_valid], mlp_sent_forecasts_lnRV[mlp_sent_valid])
else:
    mlp_sent_mse_lnRV = np.nan
    print("Warning: All MLP + VADER forecasts are NaN")

print("\nOut-of-Sample MSE (ln(RV) space):")
if not np.isnan(har_mse_lnRV):
    print(f"  HAR-Robust (Baseline):        {har_mse_lnRV:.8f} (N={har_valid.sum()})")
else:
    print(f"  HAR-Robust (Baseline):        N/A (no valid forecasts)")
if not np.isnan(har_sent_mse_lnRV):
    print(f"  HAR-Robust + VADER:            {har_sent_mse_lnRV:.8f} (N={har_sent_valid.sum()})")
else:
    print(f"  HAR-Robust + VADER:            N/A (no valid forecasts)")
if not np.isnan(mlp_base_mse_lnRV):
    print(f"  MLP (Baseline):                {mlp_base_mse_lnRV:.8f} (N={mlp_base_valid.sum()})")
else:
    print(f"  MLP (Baseline):                N/A (no valid forecasts)")
if not np.isnan(mlp_sent_mse_lnRV):
    print(f"  MLP + VADER:                   {mlp_sent_mse_lnRV:.8f} (N={mlp_sent_valid.sum()})")
else:
    print(f"  MLP + VADER:                   N/A (no valid forecasts)")

# Calculate MSE on RV scale (back-transformed)
print("\nOut-of-Sample MSE (RV space, back-transformed):")
# Get actual RV values for comparison
actual_RV_forecast = daily.iloc[start_idx:]["RV"].values

# Back-transform predictions to RV: RV = exp(lnRV)
if har_valid.sum() > 0:
    har_forecasts_RV_mse = np.exp(har_forecasts_lnRV[har_valid])
    actual_RV_valid = actual_RV_forecast[har_valid]
    har_mse_RV = mean_squared_error(actual_RV_valid, har_forecasts_RV_mse)
    print(f"  HAR-Robust (Baseline):        {har_mse_RV:.8f} (N={har_valid.sum()})")
else:
    har_mse_RV = np.nan
    print(f"  HAR-Robust (Baseline):        N/A (no valid forecasts)")

if har_sent_valid.sum() > 0:
    har_sent_forecasts_RV_mse = np.exp(har_sent_forecasts_lnRV[har_sent_valid])
    actual_RV_valid = actual_RV_forecast[har_sent_valid]
    har_sent_mse_RV = mean_squared_error(actual_RV_valid, har_sent_forecasts_RV_mse)
    print(f"  HAR-Robust + VADER:            {har_sent_mse_RV:.8f} (N={har_sent_valid.sum()})")
else:
    har_sent_mse_RV = np.nan
    print(f"  HAR-Robust + VADER:            N/A (no valid forecasts)")

if mlp_base_valid.sum() > 0:
    mlp_base_forecasts_RV_mse = np.exp(mlp_base_forecasts_lnRV[mlp_base_valid])
    actual_RV_valid = actual_RV_forecast[mlp_base_valid]
    mlp_base_mse_RV = mean_squared_error(actual_RV_valid, mlp_base_forecasts_RV_mse)
    print(f"  MLP (Baseline):                {mlp_base_mse_RV:.8f} (N={mlp_base_valid.sum()})")
else:
    mlp_base_mse_RV = np.nan
    print(f"  MLP (Baseline):                N/A (no valid forecasts)")

if mlp_sent_valid.sum() > 0:
    mlp_sent_forecasts_RV_mse = np.exp(mlp_sent_forecasts_lnRV[mlp_sent_valid])
    actual_RV_valid = actual_RV_forecast[mlp_sent_valid]
    mlp_sent_mse_RV = mean_squared_error(actual_RV_valid, mlp_sent_forecasts_RV_mse)
    print(f"  MLP + VADER:                   {mlp_sent_mse_RV:.8f} (N={mlp_sent_valid.sum()})")
else:
    mlp_sent_mse_RV = np.nan
    print(f"  MLP + VADER:                   N/A (no valid forecasts)")

# Save forecast results
forecast_results = pd.DataFrame({
    "date": forecast_dates,
    "actual_lnRV": actual_lnRV,
    "har_forecast_lnRV": har_forecasts_lnRV,
    "har_sent_forecast_lnRV": har_sent_forecasts_lnRV,
    "mlp_base_forecast_lnRV": mlp_base_forecasts_lnRV,
    "mlp_sent_forecast_lnRV": mlp_sent_forecasts_lnRV,
    "har_residual_var": har_residual_vars,
    "har_sent_residual_var": har_sent_residual_vars,
    "mlp_base_residual_var": mlp_base_residual_vars,
    "mlp_sent_residual_var": mlp_sent_residual_vars
})
forecast_output = processed_dir / "vader_forecasts.csv"
forecast_results.to_csv(forecast_output, index=False)
print(f"\n✅ Forecast results saved to: {forecast_output}")

# =====================================================
# 9. BACK-TRANSFORM FOR PLOTTING ONLY
# =====================================================
print("\n" + "="*60)
print("7. BACK-TRANSFORMING FOR VISUALIZATION")
print("="*60)

# Back-transform using: RV_hat = exp(ln_RV_hat + 0.5 * sigma_hat^2)
# where sigma_hat^2 is the residual variance from the ln(RV) model
# Handle NaN values properly
har_forecasts_RV = np.where(
    np.isnan(har_forecasts_lnRV) | np.isnan(har_residual_vars),
    np.nan,
    np.exp(har_forecasts_lnRV + 0.5 * har_residual_vars)
)
har_sent_forecasts_RV = np.where(
    np.isnan(har_sent_forecasts_lnRV) | np.isnan(har_sent_residual_vars),
    np.nan,
    np.exp(har_sent_forecasts_lnRV + 0.5 * har_sent_residual_vars)
)
mlp_base_forecasts_RV = np.where(
    np.isnan(mlp_base_forecasts_lnRV) | np.isnan(mlp_base_residual_vars),
    np.nan,
    np.exp(mlp_base_forecasts_lnRV + 0.5 * mlp_base_residual_vars)
)
mlp_sent_forecasts_RV = np.where(
    np.isnan(mlp_sent_forecasts_lnRV) | np.isnan(mlp_sent_residual_vars),
    np.nan,
    np.exp(mlp_sent_forecasts_lnRV + 0.5 * mlp_sent_residual_vars)
)

# Get actual RV values
actual_RV = daily.iloc[start_idx:]["RV"].values

# =====================================================
# 10. TIME-SERIES PLOT
# =====================================================
print("\n" + "="*60)
print("8. CREATING TIME-SERIES PLOT")
print("="*60)

# Create date index for plotting
plot_dates = pd.to_datetime(forecast_dates)

# Check how many valid forecasts we have
print(f"\nValid forecast counts:")
print(f"  HAR-Robust: {har_valid.sum()}/{len(har_forecasts_RV)}")
print(f"  HAR-Robust + VADER: {har_sent_valid.sum()}/{len(har_sent_forecasts_RV)}")
print(f"  MLP: {mlp_base_valid.sum()}/{len(mlp_base_forecasts_RV)}")
print(f"  MLP + VADER: {mlp_sent_valid.sum()}/{len(mlp_sent_forecasts_RV)}")

plt.figure(figsize=(14, 8))
plt.plot(plot_dates, actual_RV, label="Actual RV", linewidth=2, color="black")

# Plot all forecasts (will skip NaN values automatically)
# Use valid masks to filter, but plot all non-NaN values
har_rv_valid = ~np.isnan(har_forecasts_RV) & np.isfinite(har_forecasts_RV)
har_sent_rv_valid = ~np.isnan(har_sent_forecasts_RV) & np.isfinite(har_sent_forecasts_RV)
mlp_base_rv_valid = ~np.isnan(mlp_base_forecasts_RV) & np.isfinite(mlp_base_forecasts_RV)
mlp_sent_rv_valid = ~np.isnan(mlp_sent_forecasts_RV) & np.isfinite(mlp_sent_forecasts_RV)

if har_rv_valid.sum() > 0:
    plt.plot(plot_dates[har_rv_valid], har_forecasts_RV[har_rv_valid], label="HAR", linewidth=1.5, linestyle="--", color="blue", alpha=0.7)
else:
    print("  Warning: No valid HAR-Robust forecasts to plot")
    
if har_sent_rv_valid.sum() > 0:
    plt.plot(plot_dates[har_sent_rv_valid], har_sent_forecasts_RV[har_sent_rv_valid], label="HAR + Sentiment", linewidth=1.5, linestyle="--", color="red", alpha=0.7)
else:
    print("  Warning: No valid HAR-Robust + VADER forecasts to plot")
    
if mlp_base_rv_valid.sum() > 0:
    plt.plot(plot_dates[mlp_base_rv_valid], mlp_base_forecasts_RV[mlp_base_rv_valid], label="MLP", linewidth=1.5, linestyle="--", color="orange", alpha=0.7)
else:
    print("  Warning: No valid MLP forecasts to plot")
    
if mlp_sent_rv_valid.sum() > 0:
    plt.plot(plot_dates[mlp_sent_rv_valid], mlp_sent_forecasts_RV[mlp_sent_rv_valid], label="MLP + Sentiment", linewidth=1.5, linestyle="--", color="green", alpha=0.7)
else:
    print("  Warning: No valid MLP + VADER forecasts to plot")

plt.xlabel("Date", fontsize=12)
plt.ylabel("Realized Variance (RV)", fontsize=12)
plt.title("Realized Variance Forecasts: HAR vs HAR+VADER vs MLP+VADER", fontsize=14, fontweight="bold")
plt.legend(loc="upper left", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_output = processed_dir / "vader_forecast_plot.png"
plt.savefig(plot_output, dpi=300, bbox_inches="tight")
print(f"✅ Plot saved to: {plot_output}")
plt.close()

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nSummary:")
print(f"  - In-sample diagnostics: {diagnostics_output}")
print(f"  - Out-of-sample forecasts: {forecast_output}")
print(f"  - Visualization: {plot_output}")
print(f"\nNote: MSE evaluation is in ln(RV) space (no back-transformation).")
print(f"      Back-transformation used only for visualization.")

