# Bitcoin Volatility Forecasting with Sentiment Analysis

This repository contains code for forecasting Bitcoin realized volatility using sentiment analysis from Reddit comments. The project compares three sentiment analysis approaches: VADER (dictionary-based), SVM (machine learning), and LDA (topic modeling).

## Project Structure

```
/data/
    raw/          # Raw data files (excluded from repository)
    processed/    # Generated outputs and processed data

/vader/          # VADER sentiment analysis scripts
    VADER.py
    VADER_HARR.py
    VADER_MLP.py

/svm/            # SVM sentiment analysis scripts
    SVM2_Processing.py
    clean_twitter_data.py
    SVM2_Reddit_Analyse.py
    SVM2_Compare_With_VADER.py

/lda/            # LDA topic modeling scripts
    LDA_Script.py
    perplexity_test.py
    plot_topic_share.py
    lda_volatility_forecast.py

/common/         # Common utilities (if applicable)
```

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for required packages

## Data Files

The following raw data files are required in `data/raw/`:

- `Reddit_2021.csv` - Cleaned Reddit comments from 2021
- `btc_5min_2021.csv` - Bitcoin 5-minute price data for 2021
- `tweets_cleaned.csv` - Training data for SVM model 

**Note:** Raw data files are excluded from the repository due to size. See `data/raw/README.md` for details.

## Execution Order

### 1. VADER Analysis

```bash
# Generate hourly VADER sentiment
python vader/VADER_HARR.py

# Run HAR-Robust models with VADER sentiment
python vader/VADER.py

# Run MLP models with VADER sentiment
python vader/VADER_MLP.py
```

### 2. SVM Analysis

```bash
# Clean preprocesses labelled Twitter data.
python svm/clean_twitter_data.py

# Train SVM model and generate sentiment scores
python svm/SVM2_Processing.py

# Run HAR-Robust and MLP models with SVM sentiment
python svm/SVM2_Reddit_Analyse.py

# Compare SVM and VADER approaches
python svm/SVM2_Compare_With_VADER.py
```

### 3. LDA Analysis

```bash
# Test optimal number of topics
python lda/perplexity_test.py

# Run LDA topic modeling
python lda/LDA_Script.py

# Forecast volatility using LDA topics
python lda/lda_volatility_forecast.py

# Visualize topic shares over time
python lda/plot_topic_share.py
```

## Output Files

All processed outputs are saved to `data/processed/`, including:

- Model diagnostics and forecasts
- Topic time series
- Sentiment scores
- Visualization plots

## Notes

- Scripts use cross-platform relative paths and will run on Mac, Windows, or Linux
- Large datasets may require significant processing time
- Ensure sufficient disk space for output files

