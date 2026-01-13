# Adding Repository to GitHub

## Option 1: Using GitHub Desktop (Easiest)

1. **Open GitHub Desktop**
2. **File → Add Existing Repository from Local Drive**
3. **Browse** to: `/Users/patrickstar/Desktop/QTA_Project/Second_Run/Submission`
4. **Click "Add Repository"**
5. **Publish to GitHub:**
   - Click "Publish repository" button
   - Choose repository name (e.g., `bitcoin-volatility-forecasting`)
   - Choose visibility (Private/Public)
   - Click "Publish Repository"

## Option 2: Using Command Line

### Step 1: Initialize Git Repository

```bash
cd /Users/patrickstar/Desktop/QTA_Project/Second_Run/Submission
git init
```

### Step 2: Add All Files

```bash
git add .
```

### Step 3: Create Initial Commit

```bash
git commit -m "Initial commit: Bitcoin volatility forecasting with sentiment analysis"
```

### Step 4: Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `bitcoin-volatility-forecasting` (or your choice)
3. Description: "Bitcoin realized volatility forecasting using VADER, SVM, and LDA sentiment analysis"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### Step 5: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
git remote add origin https://github.com/YOUR_USERNAME/bitcoin-volatility-forecasting.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## Option 3: Using GitHub CLI (if installed)

```bash
cd /Users/patrickstar/Desktop/QTA_Project/Second_Run/Submission
git init
git add .
git commit -m "Initial commit: Bitcoin volatility forecasting with sentiment analysis"
gh repo create bitcoin-volatility-forecasting --public --source=. --remote=origin --push
```

## Important Notes

- **Large files**: The repository contains large CSV files (361MB, 1GB). GitHub has a 100MB file size limit. You may need:
  - Git LFS (Large File Storage) for files > 100MB
  - Or exclude large files and add them to `.gitignore`
  
- **Current .gitignore**: Already configured to exclude `data/raw/*` files, so large data files won't be pushed

- **First push**: May take a few minutes if there are many files

## Verify Upload

After pushing, visit: `https://github.com/YOUR_USERNAME/bitcoin-volatility-forecasting`

You should see:
- All Python scripts in organized folders
- README.md
- requirements.txt
- .gitignore
- data/raw/README.md (explaining data files are excluded)

