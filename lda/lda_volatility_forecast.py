# =====================================================
# LDA TOPIC-BASED VOLATILITY FORECASTING
# HAR-R and MLP models with topic share index
# =====================================================
"""
LDA Topic-Based Volatility Forecasting

Predicts Bitcoin volatility using:
1. HAR-Robust model (baseline and with topic index)
2. MLP model (baseline and with topic index)

Uses topic_freq_index at t-1 as additional predictor.

Usage:
    python lda_volatility_forecast.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# =====================================================
# CONFIGURATION
# =====================================================
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

output_dir = BASE_DIR / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

# Data file paths
BTC_FILE = BASE_DIR / "data" / "raw" / "btc_5min_2021.csv"
BTC_FILE_ALT = BASE_DIR / "data" / "raw" / "btc_5min_2017_2021.csv"
TOPIC_TS_FILE = output_dir / "topic_time_series.csv"

# Initial training window size (6 months ≈ 180 days)
INITIAL_WINDOW_SIZE = 180

# =====================================================
# 1. LOAD BITCOIN PRICE DATA
# =====================================================
print("="*60)
print("1. LOADING BITCOIN PRICE DATA")
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

# Filter to 2021 only (matching topic data)
print("  Filtering to calendar year 2021 only...")
btc = btc[btc["datetime"].dt.year == 2021].copy()
print(f"  - BTC data for 2021: {len(btc):,} observations")

# =====================================================
# 2. CONSTRUCT DAILY REALIZED VARIANCE
# =====================================================
print("\n" + "="*60)
print("2. CONSTRUCTING DAILY REALIZED VARIANCE")
print("="*60)

btc["date"] = btc["datetime"].dt.date
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
# 3. LOAD TOPIC TIME SERIES DATA
# =====================================================
print("\n" + "="*60)
print("3. LOADING TOPIC TIME SERIES DATA")
print("="*60)

if not TOPIC_TS_FILE.exists():
    raise FileNotFoundError(
        f"Topic time series file not found: {TOPIC_TS_FILE}\n"
        f"Please run LDA_Script.py first to generate the data."
    )

print(f"Loading from: {TOPIC_TS_FILE}")
topic_ts = pd.read_csv(TOPIC_TS_FILE)

# Parse date column
topic_ts["date"] = pd.to_datetime(topic_ts["date"]).dt.date

print(f"  - Loaded {len(topic_ts):,} days")
print(f"  - Date range: {topic_ts['date'].min()} to {topic_ts['date'].max()}")
print(f"  - Columns: {topic_ts.columns.tolist()}")

# Extract individual topic mean columns (theta_k_mean)
topic_mean_cols = [c for c in topic_ts.columns if c.endswith("_mean") and c.startswith("theta_")]
topic_mean_cols = sorted(topic_mean_cols, key=lambda x: int(x.split("_")[1]))

if len(topic_mean_cols) == 0:
    raise ValueError("No topic mean columns (theta_k_mean) found in topic time series data")

print(f"  - Found {len(topic_mean_cols)} topic mean columns: {topic_mean_cols}")

# =====================================================
# 4. MERGE BTC RV + TOPIC DATA
# =====================================================
print("\n" + "="*60)
print("4. MERGING BTC DAILY RV WITH TOPIC DATA")
print("="*60)

# Merge daily RV with topic time series (include all topic mean columns)
merge_cols = ["date"] + topic_mean_cols
daily = pd.merge(daily_rv, topic_ts[merge_cols], on="date", how="inner")
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

# Lag each topic mean by one day (t-1)
for col in topic_mean_cols:
    topic_num = col.split("_")[1]
    daily[f"{col}_lag1"] = daily[col].shift(1)

daily = daily.dropna()
daily = daily.sort_values("date").reset_index(drop=True)

print(f"Merged data: {len(daily):,} days")
print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")

# =====================================================
# 5. IN-SAMPLE MODEL ESTIMATION
# =====================================================
print("\n" + "="*60)
print("5. IN-SAMPLE MODEL ESTIMATION")
print("="*60)

# Use initial training window for in-sample diagnostics
train_end_idx = INITIAL_WINDOW_SIZE
if train_end_idx >= len(daily):
    raise ValueError(f"Initial window size ({INITIAL_WINDOW_SIZE}) exceeds available data ({len(daily)} days)")

train_data = daily.iloc[:train_end_idx].copy()
print(f"Using first {len(train_data)} days for in-sample estimation")
print(f"Training period: {train_data['date'].min()} to {train_data['date'].max()}")

# Prepare data for in-sample estimation
X_base_df = train_data[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"]].copy()
X_base_df.columns = ["ln(RV_daily_t-1)", "ln(RV_weekly_t-1)", "ln(RV_monthly_t-1)"]
X_base_const = sm.add_constant(X_base_df)

# Create topic lag columns list
topic_lag_cols = [f"{col}_lag1" for col in topic_mean_cols]
X_topic_df = train_data[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"] + topic_lag_cols].copy()
# Create column names
col_names = ["ln(RV_daily_t-1)", "ln(RV_weekly_t-1)", "ln(RV_monthly_t-1)"] + [f"topic_{col.split('_')[1]}_t-1" for col in topic_mean_cols]
X_topic_df.columns = col_names
X_topic_const = sm.add_constant(X_topic_df)

# For MLP, use numpy arrays (scaling)
X_base = train_data[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"]].values
X_topic = train_data[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"] + topic_lag_cols].values
y = train_data["lnRV"].values

# Scale features for MLP
scaler_base = StandardScaler()
scaler_topic = StandardScaler()
X_base_scaled = scaler_base.fit_transform(X_base)
X_topic_scaled = scaler_topic.fit_transform(X_topic)

# Model 1: HAR-Robust (Baseline)
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
print(f"AIC: {har_diagnostics['AIC']:.6f}")
print(f"BIC: {har_diagnostics['BIC']:.6f}")
print("\nCoefficient Statistics:")
print(har_model.summary().tables[1])

# Model 2: HAR-Robust + Topic Index
print("\n--- HAR-Robust + Topic Index ---")
har_topic_model = sm.OLS(y, X_topic_const).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
har_topic_diagnostics = {
    "Model": "HAR-Robust + Topic Index",
    "N": har_topic_model.nobs,
    "R²": har_topic_model.rsquared,
    "Adj R²": har_topic_model.rsquared_adj,
    "F-statistic": har_topic_model.fvalue,
    "F p-value": har_topic_model.f_pvalue,
    "Log-likelihood": har_topic_model.llf,
    "AIC": har_topic_model.aic,
    "BIC": har_topic_model.bic,
    "Residual variance": har_topic_model.mse_resid
}
print(f"N: {har_topic_diagnostics['N']}")
print(f"R²: {har_topic_diagnostics['R²']:.6f}")
print(f"Adj R²: {har_topic_diagnostics['Adj R²']:.6f}")
print(f"F-statistic: {har_topic_diagnostics['F-statistic']:.6f} (p-value: {har_topic_diagnostics['F p-value']:.6e})")
print(f"AIC: {har_topic_diagnostics['AIC']:.6f}")
print(f"BIC: {har_topic_diagnostics['BIC']:.6f}")
print("\nCoefficient Statistics:")
print(har_topic_model.summary().tables[1])

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
print(f"AIC: {mlp_base_diagnostics['AIC']:.6f}")
print(f"BIC: {mlp_base_diagnostics['BIC']:.6f}")

# Model 4: MLP + Topic Index
print("\n--- MLP + Topic Index ---")
mlp_topic_model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation='tanh',
    solver='adam',
    max_iter=2000,
    tol=1e-4,
    random_state=42,
    early_stopping=False
)
mlp_topic_model.fit(X_topic_scaled, y)
mlp_topic_pred = mlp_topic_model.predict(X_topic_scaled)

# Calculate diagnostics for MLP + Topic
n_topic = len(y)
ss_res_topic = np.sum((y - mlp_topic_pred) ** 2)
ss_tot_topic = np.sum((y - np.mean(y)) ** 2)
r2_topic = 1 - (ss_res_topic / ss_tot_topic)
r2_adj_topic = 1 - (1 - r2_topic) * (n_topic - 1) / (n_topic - X_topic.shape[1] - 1)
mse_topic = ss_res_topic / n_topic
log_likelihood_topic = -0.5 * n_topic * (np.log(2 * np.pi) + np.log(mse_topic) + 1)
aic_topic = -2 * log_likelihood_topic + 2 * (X_topic.shape[1] + 1)
bic_topic = -2 * log_likelihood_topic + np.log(n_topic) * (X_topic.shape[1] + 1)

mlp_topic_diagnostics = {
    "Model": "MLP + Topic Index",
    "N": n_topic,
    "R²": r2_topic,
    "Adj R²": r2_adj_topic,
    "F-statistic": np.nan,
    "F p-value": np.nan,
    "Log-likelihood": log_likelihood_topic,
    "AIC": aic_topic,
    "BIC": bic_topic,
    "Residual variance": mse_topic
}
print(f"N: {mlp_topic_diagnostics['N']}")
print(f"R²: {mlp_topic_diagnostics['R²']:.6f}")
print(f"Adj R²: {mlp_topic_diagnostics['Adj R²']:.6f}")
print(f"AIC: {mlp_topic_diagnostics['AIC']:.6f}")
print(f"BIC: {mlp_topic_diagnostics['BIC']:.6f}")

# Save in-sample diagnostics
diagnostics_df = pd.DataFrame([
    har_diagnostics,
    har_topic_diagnostics,
    mlp_base_diagnostics,
    mlp_topic_diagnostics
])
diagnostics_file = output_dir / "lda_volatility_in_sample_diagnostics.csv"
diagnostics_df.to_csv(diagnostics_file, index=False)
print(f"\n✅ Saved in-sample diagnostics to: {diagnostics_file}")

# =====================================================
# 6. OUT-OF-SAMPLE FORECASTING (EXPANDING WINDOW)
# =====================================================
print("\n" + "="*60)
print("6. OUT-OF-SAMPLE FORECASTING (EXPANDING WINDOW)")
print("="*60)

start_idx = INITIAL_WINDOW_SIZE
end_idx = len(daily)

print(f"Forecasting from day {start_idx+1} to day {end_idx}")
print(f"Total out-of-sample days: {end_idx - start_idx}")

har_forecasts_lnRV = []
har_topic_forecasts_lnRV = []
mlp_base_forecasts_lnRV = []
mlp_topic_forecasts_lnRV = []
actual_lnRV = []
forecast_dates = []

for i in range(start_idx, end_idx):
    if (i - start_idx + 1) % 50 == 0:
        print(f"  Processing day {i - start_idx + 1}/{end_idx - start_idx}...", flush=True)
    
    # Expanding window: use all data up to (but not including) day i
    train_window = daily.iloc[:i].copy()
    test_row = daily.iloc[i:i+1].copy()
    
    # Prepare training data
    X_train_base = train_window[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"]].values
    X_train_topic = train_window[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"] + topic_lag_cols].values
    y_train = train_window["lnRV"].values
    
    # Prepare test data
    X_test_base = test_row[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"]].values
    X_test_topic = test_row[["lnRV_lag1", "lnRV_lag7", "lnRV_lag30"] + topic_lag_cols].values
    y_test = test_row["lnRV"].values[0]
    
    # Scale features for MLP (refit scaler on expanding window)
    scaler_base_roll = StandardScaler()
    scaler_topic_roll = StandardScaler()
    X_train_base_scaled = scaler_base_roll.fit_transform(X_train_base)
    X_train_topic_scaled = scaler_topic_roll.fit_transform(X_train_topic)
    X_test_base_scaled = scaler_base_roll.transform(X_test_base)
    X_test_topic_scaled = scaler_topic_roll.transform(X_test_topic)
    
    # Add constant for OLS
    X_train_base_const = sm.add_constant(X_train_base, has_constant='add')
    X_test_base_const = sm.add_constant(X_test_base, has_constant='add')
    X_train_topic_const = sm.add_constant(X_train_topic, has_constant='add')
    X_test_topic_const = sm.add_constant(X_test_topic, has_constant='add')
    
    # HAR-Robust forecast
    try:
        har_model_roll = sm.OLS(y_train, X_train_base_const).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
        har_pred_lnRV = har_model_roll.predict(X_test_base_const)[0]
    except Exception as e:
        if (i - start_idx) < 5:
            print(f"    Warning: HAR-Robust forecast failed at day {i-start_idx+1}: {e}")
        har_pred_lnRV = np.nan
    
    # HAR-Robust + Topic Index forecast
    try:
        har_topic_model_roll = sm.OLS(y_train, X_train_topic_const).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
        har_topic_pred_lnRV = har_topic_model_roll.predict(X_test_topic_const)[0]
    except Exception as e:
        if (i - start_idx) < 5:
            print(f"    Warning: HAR-Robust+Topic forecast failed at day {i-start_idx+1}: {e}")
        har_topic_pred_lnRV = np.nan
    
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
    except Exception as e:
        if (i - start_idx) < 5:
            print(f"    Warning: MLP baseline forecast failed at day {i-start_idx+1}: {e}")
        mlp_base_pred_lnRV = np.nan
    
    # MLP + Topic Index forecast
    try:
        mlp_topic_model_roll = MLPRegressor(
            hidden_layer_sizes=(10,),
            activation='tanh',
            solver='adam',
            max_iter=2000,
            tol=1e-4,
            random_state=42,
            early_stopping=False
        )
        mlp_topic_model_roll.fit(X_train_topic_scaled, y_train)
        mlp_topic_pred_lnRV = mlp_topic_model_roll.predict(X_test_topic_scaled.reshape(1, -1))[0]
    except Exception as e:
        if (i - start_idx) < 5:
            print(f"    Warning: MLP+Topic forecast failed at day {i-start_idx+1}: {e}")
        mlp_topic_pred_lnRV = np.nan
    
    # Store forecasts
    har_forecasts_lnRV.append(har_pred_lnRV)
    har_topic_forecasts_lnRV.append(har_topic_pred_lnRV)
    mlp_base_forecasts_lnRV.append(mlp_base_pred_lnRV)
    mlp_topic_forecasts_lnRV.append(mlp_topic_pred_lnRV)
    actual_lnRV.append(y_test)
    forecast_dates.append(test_row["date"].values[0])

# =====================================================
# 7. OUT-OF-SAMPLE EVALUATION
# =====================================================
print("\n" + "="*60)
print("7. OUT-OF-SAMPLE FORECAST EVALUATION")
print("="*60)

# Convert to arrays and remove NaN values
har_forecasts_lnRV = np.array(har_forecasts_lnRV)
har_topic_forecasts_lnRV = np.array(har_topic_forecasts_lnRV)
mlp_base_forecasts_lnRV = np.array(mlp_base_forecasts_lnRV)
mlp_topic_forecasts_lnRV = np.array(mlp_topic_forecasts_lnRV)
actual_lnRV = np.array(actual_lnRV)

# Remove NaN values for evaluation
valid_mask = ~(np.isnan(har_forecasts_lnRV) | np.isnan(har_topic_forecasts_lnRV) | 
               np.isnan(mlp_base_forecasts_lnRV) | np.isnan(mlp_topic_forecasts_lnRV))

har_forecasts_valid = har_forecasts_lnRV[valid_mask]
har_topic_forecasts_valid = har_topic_forecasts_lnRV[valid_mask]
mlp_base_forecasts_valid = mlp_base_forecasts_lnRV[valid_mask]
mlp_topic_forecasts_valid = mlp_topic_forecasts_lnRV[valid_mask]
actual_valid = actual_lnRV[valid_mask]

# Calculate MSE in ln(RV) space
har_mse_ln = mean_squared_error(actual_valid, har_forecasts_valid)
har_topic_mse_ln = mean_squared_error(actual_valid, har_topic_forecasts_valid)
mlp_base_mse_ln = mean_squared_error(actual_valid, mlp_base_forecasts_valid)
mlp_topic_mse_ln = mean_squared_error(actual_valid, mlp_topic_forecasts_valid)

# Helper function to format numbers (scientific notation if too small)
def format_metric(value, threshold=0.001):
    if abs(value) < threshold:
        return f"{value:.4e}"
    else:
        return f"{value:.6f}"

print(f"\nOut-of-sample MSE (ln(RV) space):")
print(f"  HAR-Robust:              {format_metric(har_mse_ln)}")
print(f"  HAR-Robust + Topic:      {format_metric(har_topic_mse_ln)}")
print(f"  MLP:                     {format_metric(mlp_base_mse_ln)}")
print(f"  MLP + Topic:             {format_metric(mlp_topic_mse_ln)}")

# Calculate improvement in ln(RV) space
har_improvement_ln = ((har_mse_ln - har_topic_mse_ln) / har_mse_ln) * 100
mlp_improvement_ln = ((mlp_base_mse_ln - mlp_topic_mse_ln) / mlp_base_mse_ln) * 100

print(f"\nImprovement from adding topics (ln(RV) space):")
print(f"  HAR-Robust: {har_improvement_ln:.2f}% reduction in MSE")
print(f"  MLP:        {mlp_improvement_ln:.2f}% reduction in MSE")

# =====================================================
# EVALUATION IN REAL RV SPACE (BACK-TRANSFORMED)
# =====================================================
print("\n" + "="*60)
print("EVALUATION IN REAL RV SPACE")
print("="*60)

# Back-transform from ln(RV) to RV: RV = exp(ln(RV))
actual_RV = np.exp(actual_valid)
har_forecasts_RV = np.exp(har_forecasts_valid)
har_topic_forecasts_RV = np.exp(har_topic_forecasts_valid)
mlp_base_forecasts_RV = np.exp(mlp_base_forecasts_valid)
mlp_topic_forecasts_RV = np.exp(mlp_topic_forecasts_valid)

# Calculate MSE in RV space
har_mse_rv = mean_squared_error(actual_RV, har_forecasts_RV)
har_topic_mse_rv = mean_squared_error(actual_RV, har_topic_forecasts_RV)
mlp_base_mse_rv = mean_squared_error(actual_RV, mlp_base_forecasts_RV)
mlp_topic_mse_rv = mean_squared_error(actual_RV, mlp_topic_forecasts_RV)

# Calculate RMSE in RV space
har_rmse_rv = np.sqrt(har_mse_rv)
har_topic_rmse_rv = np.sqrt(har_topic_mse_rv)
mlp_base_rmse_rv = np.sqrt(mlp_base_mse_rv)
mlp_topic_rmse_rv = np.sqrt(mlp_topic_mse_rv)

# Calculate MAE in RV space
har_mae_rv = np.mean(np.abs(actual_RV - har_forecasts_RV))
har_topic_mae_rv = np.mean(np.abs(actual_RV - har_topic_forecasts_RV))
mlp_base_mae_rv = np.mean(np.abs(actual_RV - mlp_base_forecasts_RV))
mlp_topic_mae_rv = np.mean(np.abs(actual_RV - mlp_topic_forecasts_RV))

# Calculate MAPE (Mean Absolute Percentage Error) in RV space
har_mape_rv = np.mean(np.abs((actual_RV - har_forecasts_RV) / actual_RV)) * 100
har_topic_mape_rv = np.mean(np.abs((actual_RV - har_topic_forecasts_RV) / actual_RV)) * 100
mlp_base_mape_rv = np.mean(np.abs((actual_RV - mlp_base_forecasts_RV) / actual_RV)) * 100
mlp_topic_mape_rv = np.mean(np.abs((actual_RV - mlp_topic_forecasts_RV) / actual_RV)) * 100

print(f"\nOut-of-sample Metrics (RV space):")
print(f"\nMSE (Realized Variance):")
print(f"  HAR-Robust:              {format_metric(har_mse_rv)}")
print(f"  HAR-Robust + Topic:      {format_metric(har_topic_mse_rv)}")
print(f"  MLP:                     {format_metric(mlp_base_mse_rv)}")
print(f"  MLP + Topic:             {format_metric(mlp_topic_mse_rv)}")

print(f"\nRMSE (Root Mean Squared Error):")
print(f"  HAR-Robust:              {format_metric(har_rmse_rv)}")
print(f"  HAR-Robust + Topic:      {format_metric(har_topic_rmse_rv)}")
print(f"  MLP:                     {format_metric(mlp_base_rmse_rv)}")
print(f"  MLP + Topic:             {format_metric(mlp_topic_rmse_rv)}")

print(f"\nMAE (Mean Absolute Error):")
print(f"  HAR-Robust:              {format_metric(har_mae_rv)}")
print(f"  HAR-Robust + Topic:      {format_metric(har_topic_mae_rv)}")
print(f"  MLP:                     {format_metric(mlp_base_mae_rv)}")
print(f"  MLP + Topic:             {format_metric(mlp_topic_mae_rv)}")

print(f"\nMAPE (Mean Absolute Percentage Error, %):")
print(f"  HAR-Robust:              {har_mape_rv:.2f}%")
print(f"  HAR-Robust + Topic:      {har_topic_mape_rv:.2f}%")
print(f"  MLP:                     {mlp_base_mape_rv:.2f}%")
print(f"  MLP + Topic:             {mlp_topic_mape_rv:.2f}%")

# Calculate improvement in RV space
har_improvement_rv_mse = ((har_mse_rv - har_topic_mse_rv) / har_mse_rv) * 100
mlp_improvement_rv_mse = ((mlp_base_mse_rv - mlp_topic_mse_rv) / mlp_base_mse_rv) * 100
har_improvement_rv_mae = ((har_mae_rv - har_topic_mae_rv) / har_mae_rv) * 100
mlp_improvement_rv_mae = ((mlp_base_mae_rv - mlp_topic_mae_rv) / mlp_base_mae_rv) * 100

print(f"\nImprovement from adding topics (RV space):")
print(f"  HAR-Robust (MSE): {har_improvement_rv_mse:.2f}% reduction")
print(f"  HAR-Robust (MAE): {har_improvement_rv_mae:.2f}% reduction")
print(f"  MLP (MSE):        {mlp_improvement_rv_mse:.2f}% reduction")
print(f"  MLP (MAE):        {mlp_improvement_rv_mae:.2f}% reduction")

# Back-transform all forecasts to RV space (including NaN values for consistency)
actual_RV_all = np.exp(actual_lnRV)
har_forecasts_RV_all = np.exp(har_forecasts_lnRV)
har_topic_forecasts_RV_all = np.exp(har_topic_forecasts_lnRV)
mlp_base_forecasts_RV_all = np.exp(mlp_base_forecasts_lnRV)
mlp_topic_forecasts_RV_all = np.exp(mlp_topic_forecasts_lnRV)

# Save forecasts (both ln(RV) and RV space)
forecasts_df = pd.DataFrame({
    "date": forecast_dates,
    "actual_lnRV": actual_lnRV,
    "actual_RV": actual_RV_all,
    "har_forecast_lnRV": har_forecasts_lnRV,
    "har_forecast_RV": har_forecasts_RV_all,
    "har_topic_forecast_lnRV": har_topic_forecasts_lnRV,
    "har_topic_forecast_RV": har_topic_forecasts_RV_all,
    "mlp_forecast_lnRV": mlp_base_forecasts_lnRV,
    "mlp_forecast_RV": mlp_base_forecasts_RV_all,
    "mlp_topic_forecast_lnRV": mlp_topic_forecasts_lnRV,
    "mlp_topic_forecast_RV": mlp_topic_forecasts_RV_all
})
forecasts_file = output_dir / "lda_volatility_forecasts.csv"
forecasts_df.to_csv(forecasts_file, index=False)
print(f"\n✅ Saved forecasts to: {forecasts_file}")

# Save comparison results (both spaces)
comparison_df = pd.DataFrame({
    "Model": ["HAR-Robust", "HAR-Robust + Topic", "MLP", "MLP + Topic"],
    "MSE_lnRV": [har_mse_ln, har_topic_mse_ln, mlp_base_mse_ln, mlp_topic_mse_ln],
    "MSE_RV": [har_mse_rv, har_topic_mse_rv, mlp_base_mse_rv, mlp_topic_mse_rv],
    "RMSE_RV": [har_rmse_rv, har_topic_rmse_rv, mlp_base_rmse_rv, mlp_topic_rmse_rv],
    "MAE_RV": [har_mae_rv, har_topic_mae_rv, mlp_base_mae_rv, mlp_topic_mae_rv],
    "MAPE_RV_%": [har_mape_rv, har_topic_mape_rv, mlp_base_mape_rv, mlp_topic_mape_rv],
    "Improvement_lnRV_%": [0, har_improvement_ln, 0, mlp_improvement_ln],
    "Improvement_RV_MSE_%": [0, har_improvement_rv_mse, 0, mlp_improvement_rv_mse],
    "Improvement_RV_MAE_%": [0, har_improvement_rv_mae, 0, mlp_improvement_rv_mae]
})
comparison_file = output_dir / "lda_volatility_comparison.csv"
comparison_df.to_csv(comparison_file, index=False)
print(f"✅ Saved comparison results to: {comparison_file}")

# =====================================================
# 8. PLOT FORECASTS
# =====================================================
print("\n" + "="*60)
print("8. CREATING FORECAST PLOTS")
print("="*60)

# Plot 1: ln(RV) space
fig1, axes1 = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1a: HAR models (ln(RV))
ax1a = axes1[0]
ax1a.plot(forecast_dates, actual_lnRV, label="Actual", color="black", linewidth=2, alpha=0.7)
ax1a.plot(forecast_dates, har_forecasts_lnRV, label="HAR-Robust", linestyle="--", alpha=0.8)
ax1a.plot(forecast_dates, har_topic_forecasts_lnRV, label="HAR-Robust + Topic", linestyle="--", alpha=0.8)
ax1a.set_xlabel("Date", fontsize=12)
ax1a.set_ylabel("ln(RV)", fontsize=12)
ax1a.set_title("HAR-Robust Models: Out-of-Sample Forecasts (ln(RV) space)", fontsize=14, fontweight="bold")
ax1a.legend(fontsize=10)
ax1a.grid(True, alpha=0.3)
ax1a.tick_params(axis="x", rotation=45)

# Plot 1b: MLP models (ln(RV))
ax1b = axes1[1]
ax1b.plot(forecast_dates, actual_lnRV, label="Actual", color="black", linewidth=2, alpha=0.7)
ax1b.plot(forecast_dates, mlp_base_forecasts_lnRV, label="MLP", linestyle="--", alpha=0.8)
ax1b.plot(forecast_dates, mlp_topic_forecasts_lnRV, label="MLP + Topic", linestyle="--", alpha=0.8)
ax1b.set_xlabel("Date", fontsize=12)
ax1b.set_ylabel("ln(RV)", fontsize=12)
ax1b.set_title("MLP Models: Out-of-Sample Forecasts (ln(RV) space)", fontsize=14, fontweight="bold")
ax1b.legend(fontsize=10)
ax1b.grid(True, alpha=0.3)
ax1b.tick_params(axis="x", rotation=45)

plt.tight_layout()
plot_file_ln = output_dir / "lda_volatility_forecast_plot_lnRV.png"
plt.savefig(plot_file_ln, dpi=100, bbox_inches="tight")
print(f"✅ Saved ln(RV) forecast plot to: {plot_file_ln}")

# Plot 2: RV space
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))

# Plot 2a: HAR models (RV)
ax2a = axes2[0]
ax2a.plot(forecast_dates, actual_RV_all, label="Actual", color="black", linewidth=2, alpha=0.7)
ax2a.plot(forecast_dates, har_forecasts_RV_all, label="HAR-Robust", linestyle="--", alpha=0.8)
ax2a.plot(forecast_dates, har_topic_forecasts_RV_all, label="HAR-Robust + Topic", linestyle="--", alpha=0.8)
ax2a.set_xlabel("Date", fontsize=12)
ax2a.set_ylabel("Realized Variance (RV)", fontsize=12)
ax2a.set_title("HAR-Robust Models: Out-of-Sample Forecasts (RV space)", fontsize=14, fontweight="bold")
ax2a.legend(fontsize=10)
ax2a.grid(True, alpha=0.3)
ax2a.tick_params(axis="x", rotation=45)

# Plot 2b: MLP models (RV)
ax2b = axes2[1]
ax2b.plot(forecast_dates, actual_RV_all, label="Actual", color="black", linewidth=2, alpha=0.7)
ax2b.plot(forecast_dates, mlp_base_forecasts_RV_all, label="MLP", linestyle="--", alpha=0.8)
ax2b.plot(forecast_dates, mlp_topic_forecasts_RV_all, label="MLP + Topic", linestyle="--", alpha=0.8)
ax2b.set_xlabel("Date", fontsize=12)
ax2b.set_ylabel("Realized Variance (RV)", fontsize=12)
ax2b.set_title("MLP Models: Out-of-Sample Forecasts (RV space)", fontsize=14, fontweight="bold")
ax2b.legend(fontsize=10)
ax2b.grid(True, alpha=0.3)
ax2b.tick_params(axis="x", rotation=45)

plt.tight_layout()
plot_file_rv = output_dir / "lda_volatility_forecast_plot_RV.png"
plt.savefig(plot_file_rv, dpi=100, bbox_inches="tight")
print(f"✅ Saved RV forecast plot to: {plot_file_rv}")

# =====================================================
# SUMMARY
# =====================================================
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\n✅ In-sample diagnostics saved to: {diagnostics_file}")
print(f"✅ Out-of-sample forecasts saved to: {forecasts_file}")
print(f"✅ Model comparison saved to: {comparison_file}")
print(f"✅ Forecast plots (ln(RV)) saved to: {plot_file_ln}")
print(f"✅ Forecast plots (RV) saved to: {plot_file_rv}")
print("\nResults summary:")
print(comparison_df.to_string(index=False))

plt.show()

