# ============================================
# Clean Twitter Data: Extract date, tweet, label
# ============================================

import pandas as pd
from pathlib import Path
import re

def find_file(filename: str) -> Path:
    """Find file in common locations"""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent  # Go up from SVM2/Second_Run to project root
    
    candidates = [
        script_dir / filename,
        project_root / filename,
        project_root / "Data" / "Validated" / filename,
        Path.cwd() / filename,
        Path.cwd().parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {filename} in: " + ", ".join(str(c) for c in candidates))

def clean_text(text):
    """Clean text: remove extra whitespace, normalize"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # Remove multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print("="*60)
print("Cleaning Twitter Data")
print("="*60)

# Load data
print("\n1. Loading data...")
path = find_file("tweets_Labelled.csv")
df = pd.read_csv(path)

print(f"   Loaded: {path}")
print(f"   Original shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# Select only date, text (tweet), and sentiment_label
print("\n2. Selecting columns: date, text, sentiment_label...")
df_clean = df[['date', 'text', 'sentiment_label']].copy()

# Clean text column
print("\n3. Cleaning text...")
df_clean['text'] = df_clean['text'].apply(clean_text)

# Remove rows with empty text
print("\n4. Removing rows with empty text...")
before_count = len(df_clean)
df_clean = df_clean[df_clean['text'].str.len() > 0].copy()
after_count = len(df_clean)
print(f"   Removed {before_count - after_count} empty rows")

# Ensure date is datetime
print("\n5. Converting date to datetime...")
df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')

# Remove rows with invalid dates
print("\n6. Removing rows with invalid dates...")
before_count = len(df_clean)
df_clean = df_clean.dropna(subset=['date'])
after_count = len(df_clean)
print(f"   Removed {before_count - after_count} rows with invalid dates")

# Rename columns for clarity
df_clean = df_clean.rename(columns={
    'text': 'tweet',
    'sentiment_label': 'label'
})

# Sort by date
df_clean = df_clean.sort_values('date').reset_index(drop=True)

print("\n7. Final dataset:")
print(f"   Shape: {df_clean.shape}")
print(f"   Columns: {df_clean.columns.tolist()}")
print(f"\n   Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
print(f"\n   Label distribution:")
print(df_clean['label'].value_counts(dropna=False))

# Save cleaned file
output_path = Path(__file__).parent / "tweets_cleaned.csv"
df_clean.to_csv(output_path, index=False)

print(f"\n✅ Saved cleaned data to: {output_path}")
print(f"   Total rows: {len(df_clean):,}")

# Show sample
print("\n8. Sample of cleaned data:")
print(df_clean.head(10).to_string())

