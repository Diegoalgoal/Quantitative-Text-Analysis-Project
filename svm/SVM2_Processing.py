# ============================================
# Step 1: Load Data + Step 2: TF-IDF Vectorisation
# ============================================

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

# ============================================
# Step 3: Baseline Linear SVM
# ============================================

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

def run_baseline_svm(X, y):
    """
    Input:
        X : TF-IDF feature matrix
        y : sentiment labels
    Output:
        fitted LinearSVC model
    """

    clf = LinearSVC(
        C=1.0,
        random_state=42
    )

    clf.fit(X, y)

    y_pred = clf.predict(X)

    print("Baseline Linear SVM (in-sample)")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y, y_pred))

    return clf

# ============================================
# Step 4.5: Confusion Matrix
# ============================================

def print_confusion_matrix(y_true, y_pred, labels):
    """
    Prints a labeled confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True {l}" for l in labels],
        columns=[f"Pred {l}" for l in labels]
    )

    print("\nConfusion Matrix:")
    print(cm_df)
    return cm_df

def export_confusion_matrix_heatmap(y_true, y_pred, labels, output_path="confusion_matrix_heatmap.png"):
    """
    Export confusion matrix as a PNG heatmap
    
    Inputs:
        y_true : true labels
        y_pred : predicted labels
        labels : list of class labels
        output_path : path to save the PNG file
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True {l}" for l in labels],
        columns=[f"Pred {l}" for l in labels]
    )
    
    # Create heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, label='Count')
    
    # Set ticks and labels
    ax.set(xticks=np.arange(len(labels)),
           yticks=np.arange(len(labels)),
           xticklabels=[f"Pred {l}" for l in labels],
           yticklabels=[f"True {l}" for l in labels],
           title='Confusion Matrix Heatmap',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    # Add grid lines
    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    
    # Save to file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Confusion matrix heatmap saved to: {output_path}")
    return cm_df

# ============================================
# Step 4.6: Visualize SVM in 3D
# ============================================

def plot_svm_2d(X, y):
    # 1. Reduce dimensionality to 3D
    svd = TruncatedSVD(n_components=3, random_state=42)
    X_3d = svd.fit_transform(X)

    # 2. Train SVM in 3D (visualisation only)
    clf = LinearSVC(C=1.0, random_state=42)
    clf.fit(X_3d, y)

    # 3. Create label mapping
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    y_numeric = y.map(label_map)

    # 4. Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 5. Plot data points colored by class
    colors_map = {0: '#FF0000', 1: '#0000FF', 2: '#00AA00'}
    labels_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    for label_num in [0, 1, 2]:
        mask = y_numeric == label_num
        ax.scatter(
            X_3d[mask, 0], 
            X_3d[mask, 1], 
            X_3d[mask, 2],
            c=colors_map[label_num],
            label=labels_map[label_num],
            s=10,
            alpha=0.6,
            edgecolors='k',
            linewidth=0.1
        )
    
    # 6. Draw hyperplanes (decision boundaries)
    classes = clf.classes_
    coef = clf.coef_
    intercept = clf.intercept_
    
    # Get ranges for mesh
    x_min, x_max = X_3d[:, 0].min() - 0.5, X_3d[:, 0].max() + 0.5
    y_min, y_max = X_3d[:, 1].min() - 0.5, X_3d[:, 1].max() + 0.5
    z_min, z_max = X_3d[:, 2].min() - 0.5, X_3d[:, 2].max() + 0.5
    
    # Create mesh for hyperplane visualization
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 50),
        np.linspace(y_min, y_max, 50)
    )
    
    # Draw decision boundaries between each pair of classes
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            # Decision boundary plane: (coef[i] - coef[j]) · x + (intercept[i] - intercept[j]) = 0
            coef_diff = coef[i] - coef[j]
            intercept_diff = intercept[i] - intercept[j]
            
            # Solve for z: coef_diff[0]*x + coef_diff[1]*y + coef_diff[2]*z + intercept_diff = 0
            # z = -(coef_diff[0]*x + coef_diff[1]*y + intercept_diff) / coef_diff[2]
            if abs(coef_diff[2]) > 1e-6:
                zz = -(coef_diff[0] * xx + coef_diff[1] * yy + intercept_diff) / coef_diff[2]
                # Clip to visible range
                zz = np.clip(zz, z_min, z_max)
                ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray', linewidth=0, antialiased=True)
    
    # Set labels and title
    ax.set_xlabel('SVD Component 1', fontsize=12)
    ax.set_ylabel('SVD Component 2', fontsize=12)
    ax.set_zlabel('SVD Component 3', fontsize=12)
    ax.set_title('3D Projection of TF-IDF Space with Linear SVM\n(Hyperplanes Shown)', fontsize=14)
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()

# ============================================
# Step 4: Linear SVM with K-Fold CV
# ============================================

import numpy as np
from sklearn.model_selection import StratifiedKFold

def run_svm_kfold(X, y, n_splits=5, return_first_fold_predictions=False):
    """
    Linear SVM with Stratified K-Fold Cross-Validation
    
    NOTE: This function evaluates a FIXED hyperparameter (C=1.0) across folds.
    For proper model selection with hyperparameter tuning, use run_svm_gridsearch() instead.

    Inputs:
        X : TF-IDF feature matrix
        y : sentiment labels
        n_splits : number of folds
        return_first_fold_predictions : if True, returns predictions from first fold for confusion matrix

    Output:
        list of fold accuracies, or (accuracies, y_test, y_pred) if return_first_fold_predictions=True
    """

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    accuracies = []
    first_fold_predictions = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = LinearSVC(C=1.0, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"Fold {fold} accuracy: {acc:.4f}")
        
        # Save predictions from first fold for confusion matrix
        if fold == 1 and return_first_fold_predictions:
            first_fold_predictions = {
                'y_test': y_test,
                'y_pred': y_pred,
                'clf': clf  # Return the trained model from first fold
            }

    print("\nMean CV accuracy:", np.mean(accuracies))
    print("Std CV accuracy:", np.std(accuracies))

    if return_first_fold_predictions:
        return accuracies, first_fold_predictions
    return accuracies

def run_svm_hyperparameter_tuning(X, y, n_splits=5, output_dir=None):
    """
    Proper hyperparameter tuning workflow:
    (1) Define hyperparameter grid
    (2) Run k-fold CV on training data to select best hyperparameters
    (3) Retrain final model on full dataset with optimal hyperparameters
    
    Inputs:
        X : TF-IDF feature matrix (sparse or dense)
        y : sentiment labels
        n_splits : number of folds for cross-validation
        output_dir : directory to save CV results (Path object)
    
    Output:
        dict with keys:
            - 'best_params': optimal hyperparameters
            - 'best_cv_score': best cross-validated score (mean)
            - 'cv_results': DataFrame with all CV results (mean, std per hyperparameter)
            - 'final_model': SVM model retrained on full dataset with best hyperparameters
            - 'grid_search': GridSearchCV object
    """
    from sklearn.model_selection import StratifiedKFold
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING WORKFLOW")
    print("="*60)
    
    # ============================================
    # Step 1: Define hyperparameter grid
    # ============================================
    print("\n[Step 1] Defining hyperparameter grid...")
    param_grid = {
        'C': [0.1, 1.0, 10.0]  # Test different regularization strengths
    }
    print(f"   Hyperparameter grid: {param_grid}")
    print(f"   Total hyperparameter combinations: {len(param_grid['C'])}")
    
    # ============================================
    # Step 2: Run k-fold CV to select best hyperparameters
    # ============================================
    print(f"\n[Step 2] Running {n_splits}-fold cross-validation on training data...")
    print(f"   This will evaluate each hyperparameter combination via k-fold CV")
    print(f"   Total model fits: {len(param_grid['C'])} combinations × {n_splits} folds = {len(param_grid['C']) * n_splits} fits")
    
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )
    
    # Create base estimator
    base_clf = LinearSVC(random_state=42, max_iter=2000)
    
    # GridSearchCV with k-fold CV
    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,  # Use all available cores
        verbose=1,
        return_train_score=True  # Return train scores for analysis
    )
    
    # Fit GridSearchCV (runs k-fold CV for each hyperparameter combination)
    grid_search.fit(X, y)
    
    print(f"\n✅ K-fold CV complete!")
    print(f"\nBest hyperparameters (selected based on mean validation score):")
    print(f"   {grid_search.best_params_}")
    print(f"Best mean cross-validated accuracy: {grid_search.best_score_:.4f}")
    
    # Extract CV results for reporting
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Create summary of CV results (mean, std per hyperparameter)
    cv_summary = cv_results_df[[
        'param_C', 
        'mean_test_score', 
        'std_test_score',
        'mean_train_score',
        'std_train_score'
    ]].copy()
    cv_summary.columns = ['C', 'Mean_Val_Score', 'Std_Val_Score', 'Mean_Train_Score', 'Std_Train_Score']
    cv_summary = cv_summary.sort_values('Mean_Val_Score', ascending=False)
    
    print(f"\nCross-Validation Results Summary:")
    print(cv_summary.to_string(index=False))
    
    # Save CV results to file
    if output_dir is not None:
        cv_results_path = output_dir / "svm_cv_results.csv"
        cv_summary.to_csv(cv_results_path, index=False)
        print(f"\n✅ CV results saved to: {cv_results_path}")
        
        # Also save full CV results
        cv_results_full_path = output_dir / "svm_cv_results_full.csv"
        cv_results_df.to_csv(cv_results_full_path, index=False)
        print(f"✅ Full CV results saved to: {cv_results_full_path}")
    
    # ============================================
    # Step 3: Retrain final model on full dataset with optimal hyperparameters
    # ============================================
    print(f"\n[Step 3] Retraining final SVM model on FULL dataset...")
    print(f"   Using optimal hyperparameters: {grid_search.best_params_}")
    
    final_model = LinearSVC(
        C=grid_search.best_params_['C'],
        random_state=42,
        max_iter=2000
    )
    
    # Retrain on full dataset
    final_model.fit(X, y)
    
    # Get number of samples (handle sparse matrices)
    n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
    print(f"✅ Final model retrained on full dataset ({n_samples:,} samples)")
    print(f"   This is the model that will be used for predictions")
    
    # Prepare return dictionary
    results = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'best_cv_std': cv_summary.iloc[0]['Std_Val_Score'],
        'cv_results': cv_summary,
        'cv_results_full': cv_results_df,
        'final_model': final_model,  # Model retrained on full dataset
        'grid_search': grid_search
    }
    
    return results

# ============================================
# Step 6: Extract Top Words by SVM Weights
# ============================================

def print_top_words_by_svm_weights(clf, vectorizer, n_words=20):
    """
    Print words with highest SVM weights for each class.
    
    Inputs:
        clf : fitted LinearSVC model
        vectorizer : fitted TfidfVectorizer
        n_words : number of top words to show per class
    """
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients (weights) - shape: (n_classes, n_features)
    coefficients = clf.coef_
    classes = clf.classes_
    
    print("="*60)
    print(f"Top {n_words} Words by SVM Weights (per class)")
    print("="*60)
    
    for i, class_name in enumerate(classes):
        # Get weights for this class
        class_weights = coefficients[i]
        
        # Get indices sorted by absolute weight (descending)
        top_indices = np.argsort(np.abs(class_weights))[-n_words:][::-1]
        
        print(f"\n{class_name} class:")
        print("-" * 60)
        print(f"{'Word':<25} {'Weight':>15}")
        print("-" * 60)
        
        for idx in top_indices:
            word = feature_names[idx]
            weight = class_weights[idx]
            print(f"{word:<25} {weight:>15.6f}")
    
    print("\n" + "="*60)
    print("Note: Positive weights push toward this class,")
    print("      Negative weights push away from this class")
    print("="*60)

def find_file(filename: str) -> Path:
    # Get script directory (BASE_DIR)
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # searches common locations relative to BASE_DIR
    candidates = [
        BASE_DIR / filename,  # Same directory as script
        BASE_DIR / "data" / "raw" / filename,  # Data directory
        Path.cwd() / filename,
        Path.cwd() / "data" / "raw" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {filename} in: " + ", ".join(str(c) for c in candidates))

def vectorize_text(df):
    """
    Input:
        df with columns ['tweet', 'label']
    Output:
        X: TF-IDF matrix
        y: labels
        vectorizer: fitted TF-IDF object
    """
    X_text = df["tweet"].astype(str)
    y = df["label"]

    vectorizer = TfidfVectorizer(
        min_df=5,            # keeps vocabulary manageable
        max_df=0.9,
        stop_words="english",
        ngram_range=(1, 1)    # IMPORTANT: unigrams only (Lecture 5)
    )

    X = vectorizer.fit_transform(X_text)

    print("TF-IDF matrix shape:", X.shape)

    return X, y, vectorizer

# ============================================
# Main execution
# ============================================
print("="*60)
print("Step 1: Loading data...")
print("="*60)

path = find_file("tweets_cleaned.csv")
df = pd.read_csv(path)

print("Loaded:", path)
print("Rows, Cols:", df.shape)
print("Columns:", df.columns.tolist())
print(df[["tweet", "label"]].head())
print("Label counts:\n", df["label"].value_counts(dropna=False))

print("\n" + "="*60)
print("Step 2: TF-IDF Vectorization...")
print("="*60)

# Clean and prepare data
df = df[['tweet', 'label']].dropna()
df['tweet'] = df['tweet'].astype(str)

print(f"Processing {len(df):,} texts...")

# Vectorize
X, y, vectorizer = vectorize_text(df)

print(f"\n✅ Vectorization complete!")
print(f"   Features: {X.shape[1]:,}")
print(f"   Samples: {X.shape[0]:,}")

# Save outputs
BASE_DIR = Path(__file__).resolve().parent.parent
output_dir = BASE_DIR / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\n" + "="*60)
print("Step 3: Saving outputs...")
print("="*60)

# Save TF-IDF matrix (sparse)
X_path = output_dir / "X_tfidf.npz"
sparse.save_npz(X_path, X)
print(f"✅ Saved TF-IDF matrix to: {X_path}")

# Save labels
y_path = output_dir / "y_labels.csv"
y_df = pd.DataFrame({'label': y})
y_df.to_csv(y_path, index=False)
print(f"✅ Saved labels to: {y_path}")

# Save vectorizer
vectorizer_path = output_dir / "vectorizer.pkl"
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"✅ Saved vectorizer to: {vectorizer_path}")

print(f"\n📁 All outputs saved to: {output_dir}")

# ============================================
# Step 4: Run Baseline Linear SVM
# ============================================
print(f"\n" + "="*60)
print("Step 4: Running Baseline Linear SVM...")
print("="*60)

clf = run_baseline_svm(X, y)

# Save trained SVM model for later use (trained on whole dataset with fixed C=1.0)
# NOTE: This will be replaced with best model from GridSearchCV below
svm_model_path = output_dir / "svm_model.pkl"
with open(svm_model_path, 'wb') as f:
    pickle.dump(clf, f)
print(f"✅ Saved baseline SVM model (C=1.0) to: {svm_model_path}")

print(f"\n✅ Baseline SVM training complete!")

# ============================================
# Step 5: Run Linear SVM with K-Fold CV (Fixed Hyperparameter - Reporting Only)
# ============================================
print(f"\n" + "="*60)
print("Step 5a: Running Linear SVM with K-Fold CV (Fixed C=1.0)...")
print("="*60)
print("⚠️  NOTE: This evaluates a FIXED hyperparameter (C=1.0) across folds.")
print("    This is for reporting only, NOT proper model selection.")

cv_accuracies, fold1_predictions = run_svm_kfold(X, y, n_splits=5, return_first_fold_predictions=True)

print(f"\n✅ K-Fold CV (fixed hyperparameter) complete!")

# ============================================
# Step 5b: Hyperparameter Tuning with k-fold CV
# ============================================
# This follows the proper workflow:
# (1) Define hyperparameter grid
# (2) Run k-fold CV on training data to select best hyperparameters
# (3) Retrain final model on full dataset with optimal hyperparameters

tuning_results = run_svm_hyperparameter_tuning(X, y, n_splits=5, output_dir=output_dir)

# Extract results
best_params = tuning_results['best_params']
best_cv_score = tuning_results['best_cv_score']
best_cv_std = tuning_results['best_cv_std']
final_model = tuning_results['final_model']  # Model retrained on FULL dataset
cv_results = tuning_results['cv_results']

print(f"\n" + "="*60)
print("HYPERPARAMETER TUNING SUMMARY")
print("="*60)
print(f"Best hyperparameters: {best_params}")
print(f"Best CV score (mean): {best_cv_score:.4f} ± {best_cv_std:.4f}")
# Get number of samples (handle sparse matrices)
n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
print(f"Final model: Retrained on full dataset ({n_samples:,} samples)")

# Save final model (retrained on full dataset) for production use
svm_model_path = output_dir / "svm_model.pkl"
with open(svm_model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"\n✅ Final SVM model saved to: {svm_model_path}")
print(f"   Note: This model is retrained on FULL dataset with optimal hyperparameters")
print(f"   This is the model that should be used for all predictions")

# Get predictions from first fold for confusion matrix (using best hyperparameters)
print(f"\nGenerating out-of-sample predictions for confusion matrix...")
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = list(skf.split(X, y))[0]
X_train_fold1, X_test_fold1 = X[train_idx], X[test_idx]
y_train_fold1, y_test_fold1 = y.iloc[train_idx], y.iloc[test_idx]

# Train model with best hyperparameters on fold 1 training data
best_clf_fold1 = LinearSVC(C=best_params['C'], random_state=42, max_iter=2000)
best_clf_fold1.fit(X_train_fold1, y_train_fold1)
y_pred_fold1 = best_clf_fold1.predict(X_test_fold1)

fold1_predictions = {
    'y_test': y_test_fold1,
    'y_pred': y_pred_fold1,
    'clf': best_clf_fold1,
    'best_params': best_params,
    'best_cv_score': best_cv_score
}

# ============================================
# Step 4.5: Generate Confusion Matrix from K-Fold Out-of-Sample Data
# ============================================
print(f"\n" + "="*60)
print("Step 4.5: Generating Confusion Matrix from K-Fold Out-of-Sample Data...")
print("="*60)
print("Using predictions from BEST MODEL (selected via k-fold CV hyperparameter tuning)")
print(f"Best hyperparameters: {best_params}")
print(f"Best CV score: {best_cv_score:.4f} ± {best_cv_std:.4f}")
print("Using predictions from Fold 1 validation set (out-of-sample)")
print("Note: Final model used for production is retrained on FULL dataset")

# Use out-of-sample predictions from best model (first fold)
y_test_outsample = fold1_predictions['y_test']
y_pred_outsample = fold1_predictions['y_pred']
clf_fold1 = fold1_predictions['clf']

# Print confusion matrix (out-of-sample)
print(f"\nOut-of-Sample Confusion Matrix (Fold 1 validation set):")
print(f"Validation set size: {len(y_test_outsample):,} samples")
print_confusion_matrix(y_test_outsample, y_pred_outsample, labels=clf_fold1.classes_)

# Export confusion matrix as PNG heatmap (out-of-sample)
BASE_DIR = Path(__file__).resolve().parent.parent
output_dir = BASE_DIR / "data" / "processed"
cm_heatmap_path = output_dir / "confusion_matrix_heatmap.png"
export_confusion_matrix_heatmap(y_test_outsample, y_pred_outsample, labels=clf_fold1.classes_, 
                                output_path=str(cm_heatmap_path))

print(f"\n✅ Out-of-sample confusion matrix saved to: {cm_heatmap_path}")
print(f"\n" + "="*60)
print("MODEL SELECTION WORKFLOW SUMMARY")
print("="*60)
print(f"1. Hyperparameter grid defined: C ∈ {[0.1, 1.0, 10.0]}")
print(f"2. K-fold CV ({5} folds) used to evaluate each hyperparameter combination")
print(f"3. Best hyperparameters selected: {best_params} (CV score: {best_cv_score:.4f} ± {best_cv_std:.4f})")
n_samples_final = X.shape[0] if hasattr(X, 'shape') else len(X)
print(f"4. Final model retrained on FULL dataset ({n_samples_final:,} samples) with optimal hyperparameters")
print(f"5. Final model saved to: {svm_model_path}")
print(f"6. CV results saved to: {output_dir / 'svm_cv_results.csv'}")
print(f"\n✅ Confusion matrix uses:")
print(f"   - Model with best hyperparameters ({best_params})")
print(f"   - Validation set from Fold 1 (out-of-sample)")
print(f"   - Provides realistic performance estimate")

# ============================================
# Step 6: Print Top Words by SVM Weights
# ============================================
print(f"\n" + "="*60)
print("Step 6: Extracting Top Words by SVM Weights...")
print("="*60)
print(f"Using FINAL MODEL (with optimal hyperparameters: {best_params})")

print_top_words_by_svm_weights(final_model, vectorizer, n_words=20)

# ============================================
# Step 7: Apply SVM to Reddit Data
# ============================================
print(f"\n" + "="*60)
print("Step 7: Applying SVM to Reddit Data...")
print("="*60)

# Try to find Reddit CSV file - use same file as VADER.py
# Check multiple possible locations
BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data" / "raw"

# Priority order: check data directory
reddit_candidates = [
    data_dir / "Reddit_2021.csv",  # Primary location
]

reddit_file = None
for candidate_path in reddit_candidates:
    if candidate_path.exists():
        reddit_file = candidate_path
        break

# Fallback to find_file if still not found
if not reddit_file:
    for candidate_path in reddit_candidates:
        candidate_name = candidate_path.name
        try:
            reddit_file = find_file(candidate_name)
            if reddit_file.exists():
                break
        except FileNotFoundError:
            continue

if reddit_file and reddit_file.exists():
    print(f"\n1. Loading Reddit data from: {reddit_file}")
    reddit = pd.read_csv(reddit_file)
    
    print(f"   Columns found: {reddit.columns.tolist()}")
    
    # Check required columns - handle different column names
    # Map common column names to 'text'
    text_col = None
    for col_name in ["text", "clean_text", "body", "comment", "content", "message"]:
        if col_name in reddit.columns:
            text_col = col_name
            break
    
    if text_col:
        if text_col != "text":
            reddit["text"] = reddit[text_col]
            print(f"   Mapped column '{text_col}' to 'text'")
    else:
        print(f"   ⚠️  Warning: No text column found. Available columns: {reddit.columns.tolist()}")
        print(f"   Skipping Reddit analysis.")
        text_col = None
    
    # Handle date column - check for 'hour' column (YYYY-MM-DDHH format)
    if "date" not in reddit.columns:
        date_col = None
        for col_name in ["date", "hour", "created_utc", "created_at", "timestamp", "time"]:
            if col_name in reddit.columns:
                date_col = col_name
                break
        
        if date_col:
            if date_col == "hour":
                # Handle YYYY-MM-DDHH format (like "2021-01-0100")
                try:
                    reddit["date"] = pd.to_datetime(reddit[date_col], format='%Y-%m-%d%H', errors='coerce')
                except:
                    reddit["date"] = pd.to_datetime(reddit[date_col], errors='coerce')
            elif date_col == "created_utc":
                # Try both Unix timestamp and datetime string
                try:
                    reddit["date"] = pd.to_datetime(reddit[date_col], errors='coerce', unit='s')
                except:
                    reddit["date"] = pd.to_datetime(reddit[date_col], errors='coerce')
            else:
                reddit["date"] = pd.to_datetime(reddit[date_col], errors='coerce')
            print(f"   Mapped column '{date_col}' to 'date'")
    
    # Check if we have the required text column
    if text_col is None or "text" not in reddit.columns:
        print(f"   ⚠️  Warning: Cannot find text column. Skipping Reddit analysis.")
    else:
        # Clean Reddit data
        print("2. Cleaning Reddit data...")
        reddit = reddit.dropna(subset=["text"])
        reddit["text"] = reddit["text"].astype(str)
        
        if "date" in reddit.columns:
            reddit["date"] = pd.to_datetime(reddit["date"], errors='coerce')
            reddit = reddit.dropna(subset=["date"])
        else:
            # If no date column, create a dummy one
            print("   ⚠️  No date column found. Using index as date.")
            reddit["date"] = pd.Timestamp.now()
        
        print(f"   Processed {len(reddit):,} Reddit comments")
        
        # Vectorize Reddit text
        print("3. Vectorizing Reddit text...")
        X_reddit = vectorizer.transform(reddit["text"])
        print(f"   Vectorized shape: {X_reddit.shape}")
        
        # SVM predictions using FINAL MODEL (retrained on full dataset with optimal hyperparameters)
        print("4. Getting SVM predictions using FINAL MODEL...")
        print(f"   Using model with optimal hyperparameters: {best_params}")
        reddit["svm_class"] = final_model.predict(X_reddit)
        
        # Decision function: distances to hyperplanes
        decision_scores = final_model.decision_function(X_reddit)
        
        # Map columns correctly
        class_order = final_model.classes_
        score_cols = [f"score_{c}" for c in class_order]
        scores_df = pd.DataFrame(decision_scores, columns=score_cols)
        reddit = pd.concat([reddit.reset_index(drop=True), scores_df], axis=1)
        
        # Continuous sentiment (distance from Neutral)
        print("5. Calculating continuous sentiment...")
        reddit["sentiment_continuous"] = (
            reddit["score_Positive"] - reddit["score_Negative"]
        )
        reddit["sentiment_strength"] = reddit["sentiment_continuous"].abs()
        
        # Disagreement (tweet-level)
        print("6. Calculating disagreement...")
        reddit["disagreement_tweet"] = (
            reddit[["score_Negative", "score_Neutral", "score_Positive"]]
            .std(axis=1)
        )
        
        # Aggregate to daily level (if date column exists)
        if "date" in reddit.columns and reddit["date"].notna().any():
            print("7. Aggregating to daily level...")
            daily = reddit.groupby(reddit["date"].dt.date).agg(
                sentiment_mean=("sentiment_continuous", "mean"),
                sentiment_dispersion=("sentiment_continuous", "std"),
                disagreement=("disagreement_tweet", "mean"),
                sentiment_strength=("sentiment_strength", "mean"),
                n_posts=("sentiment_continuous", "count")
            ).reset_index()
            
            print("\nDaily aggregated sentiment:")
            print(daily.head(10))
            
            # Save results
            daily_output_path = output_dir / "reddit_daily_sentiment_svm.csv"
            daily.to_csv(daily_output_path, index=False)
            print(f"\n✅ Saved daily sentiment to: {daily_output_path}")
        
        # Also save full Reddit results (optional, can be large)
        reddit_output_path = output_dir / "reddit_sentiment_svm_full.csv"
        reddit.to_csv(reddit_output_path, index=False)
        print(f"✅ Saved full Reddit results to: {reddit_output_path}")
        
        print(f"\n✅ Reddit sentiment analysis complete!")
else:
    print(f"\n⚠️  Reddit file not found. Skipping Reddit analysis.")
    print(f"   Searched for: {', '.join(reddit_candidates)}")

# ============================================
# Step 8: Compare OOS RV MSE: HAR-R+VADER vs HAR-R+SVM vs MLP+VADER vs MLP+SVM
# ============================================
print(f"\n" + "="*60)
print("Step 8: Comparing Out-of-Sample RV MSE")
print("="*60)
print("Comparing: HAR-R + VADER vs HAR-R + SVM vs MLP + VADER vs MLP + SVM")

import os
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error

# Find the forecast results file from SVM2_VADER_Combined.py
BASE_DIR = Path(__file__).resolve().parent.parent
output_dir = BASE_DIR / "data" / "processed"

# Look for forecast results CSV
forecast_candidates = [
    output_dir / "svm2_vader_combined_forecasts.csv",
]

forecast_file = None
for candidate in forecast_candidates:
    if candidate.exists():
        forecast_file = candidate
        break

if forecast_file and forecast_file.exists():
    print(f"\n1. Loading forecast results from: {forecast_file}")
    forecast_df = pd.read_csv(forecast_file)
    
    # Check required columns
    required_cols = [
        'actual_lnRV',
        'har_vader_forecast_lnRV',
        'har_svm_forecast_lnRV',
        'mlp_vader_forecast_lnRV',
        'mlp_svm_forecast_lnRV',
        'har_vader_residual_var',
        'har_svm_residual_var',
        'mlp_vader_residual_var',
        'mlp_svm_residual_var'
    ]
    
    missing_cols = [col for col in required_cols if col not in forecast_df.columns]
    if missing_cols:
        print(f"   ⚠️  Warning: Missing columns: {missing_cols}")
        print(f"   Available columns: {forecast_df.columns.tolist()}")
        print(f"   Skipping comparison.")
    else:
        print(f"   ✅ Loaded {len(forecast_df):,} forecast observations")
        
        # Get actual RV by back-transforming from lnRV
        # Actual RV = exp(actual_lnRV)
        actual_RV = np.exp(forecast_df['actual_lnRV'].values)
        
        # Back-transform forecasts to RV space
        # RV_hat = exp(ln_RV_hat + 0.5 * sigma_hat^2)
        har_vader_forecast_RV = np.where(
            np.isnan(forecast_df['har_vader_forecast_lnRV']) | np.isnan(forecast_df['har_vader_residual_var']),
            np.nan,
            np.exp(forecast_df['har_vader_forecast_lnRV'] + 0.5 * forecast_df['har_vader_residual_var'])
        )
        
        har_svm_forecast_RV = np.where(
            np.isnan(forecast_df['har_svm_forecast_lnRV']) | np.isnan(forecast_df['har_svm_residual_var']),
            np.nan,
            np.exp(forecast_df['har_svm_forecast_lnRV'] + 0.5 * forecast_df['har_svm_residual_var'])
        )
        
        mlp_vader_forecast_RV = np.where(
            np.isnan(forecast_df['mlp_vader_forecast_lnRV']) | np.isnan(forecast_df['mlp_vader_residual_var']),
            np.nan,
            np.exp(forecast_df['mlp_vader_forecast_lnRV'] + 0.5 * forecast_df['mlp_vader_residual_var'])
        )
        
        mlp_svm_forecast_RV = np.where(
            np.isnan(forecast_df['mlp_svm_forecast_lnRV']) | np.isnan(forecast_df['mlp_svm_residual_var']),
            np.nan,
            np.exp(forecast_df['mlp_svm_forecast_lnRV'] + 0.5 * forecast_df['mlp_svm_residual_var'])
        )
        
        # Filter valid predictions
        har_vader_valid = ~np.isnan(har_vader_forecast_RV)
        har_svm_valid = ~np.isnan(har_svm_forecast_RV)
        mlp_vader_valid = ~np.isnan(mlp_vader_forecast_RV)
        mlp_svm_valid = ~np.isnan(mlp_svm_forecast_RV)
        
        # Calculate OOS RV MSE
        print("\n2. Calculating Out-of-Sample RV MSE...")
        
        def calc_rv_mse(actual, forecast, valid_mask):
            if valid_mask.sum() > 0:
                return mean_squared_error(actual[valid_mask], forecast[valid_mask])
            return np.nan
        
        har_vader_mse_RV = calc_rv_mse(actual_RV, har_vader_forecast_RV, har_vader_valid)
        har_svm_mse_RV = calc_rv_mse(actual_RV, har_svm_forecast_RV, har_svm_valid)
        mlp_vader_mse_RV = calc_rv_mse(actual_RV, mlp_vader_forecast_RV, mlp_vader_valid)
        mlp_svm_mse_RV = calc_rv_mse(actual_RV, mlp_svm_forecast_RV, mlp_svm_valid)
        
        # Display results
        print("\n" + "="*60)
        print("OUT-OF-SAMPLE RV MSE COMPARISON")
        print("="*60)
        print(f"\n{'Model':<25} {'RV MSE':>15} {'N':>10}")
        print("-" * 50)
        
        results = []
        if not np.isnan(har_vader_mse_RV):
            print(f"{'HAR-R + VADER':<25} {har_vader_mse_RV:>15.8f} {har_vader_valid.sum():>10}")
            results.append(("HAR-R + VADER", har_vader_mse_RV, har_vader_valid.sum()))
        else:
            print(f"{'HAR-R + VADER':<25} {'N/A':>15} {har_vader_valid.sum():>10}")
        
        if not np.isnan(har_svm_mse_RV):
            print(f"{'HAR-R + SVM':<25} {har_svm_mse_RV:>15.8f} {har_svm_valid.sum():>10}")
            results.append(("HAR-R + SVM", har_svm_mse_RV, har_svm_valid.sum()))
        else:
            print(f"{'HAR-R + SVM':<25} {'N/A':>15} {har_svm_valid.sum():>10}")
        
        if not np.isnan(mlp_vader_mse_RV):
            print(f"{'MLP + VADER':<25} {mlp_vader_mse_RV:>15.8f} {mlp_vader_valid.sum():>10}")
            results.append(("MLP + VADER", mlp_vader_mse_RV, mlp_vader_valid.sum()))
        else:
            print(f"{'MLP + VADER':<25} {'N/A':>15} {mlp_vader_valid.sum():>10}")
        
        if not np.isnan(mlp_svm_mse_RV):
            print(f"{'MLP + SVM':<25} {mlp_svm_mse_RV:>15.8f} {mlp_svm_valid.sum():>10}")
            results.append(("MLP + SVM", mlp_svm_mse_RV, mlp_svm_valid.sum()))
        else:
            print(f"{'MLP + SVM':<25} {'N/A':>15} {mlp_svm_valid.sum():>10}")
        
        # Rank models
        if results:
            print("\n" + "="*60)
            print("MODEL RANKINGS (by Out-of-Sample RV MSE, lower is better)")
            print("="*60)
            results_sorted = sorted(results, key=lambda x: x[1])
            for rank, (model, mse, n) in enumerate(results_sorted, 1):
                print(f"{rank}. {model:<25} {mse:>15.8f} (N={n})")
            
            # Calculate relative improvement
            if len(results_sorted) > 0:
                best_mse = results_sorted[0][1]
                print("\n" + "="*60)
                print("RELATIVE PERFORMANCE (vs Best Model)")
                print("="*60)
                for model, mse, n in results_sorted:
                    improvement = ((mse - best_mse) / best_mse) * 100
                    if improvement == 0:
                        print(f"{model:<25} {improvement:>15.2f}% (Best)")
                    else:
                        print(f"{model:<25} {improvement:>15.2f}% worse")
        
        # Save comparison results
        comparison_df = pd.DataFrame({
            'Model': ['HAR-R + VADER', 'HAR-R + SVM', 'MLP + VADER', 'MLP + SVM'],
            'Out_Sample_MSE_RV': [
                har_vader_mse_RV if not np.isnan(har_vader_mse_RV) else np.nan,
                har_svm_mse_RV if not np.isnan(har_svm_mse_RV) else np.nan,
                mlp_vader_mse_RV if not np.isnan(mlp_vader_mse_RV) else np.nan,
                mlp_svm_mse_RV if not np.isnan(mlp_svm_mse_RV) else np.nan
            ],
            'Valid_Forecasts': [
                har_vader_valid.sum(),
                har_svm_valid.sum(),
                mlp_vader_valid.sum(),
                mlp_svm_valid.sum()
            ]
        })
        
        comparison_output = svm2_output_dir / "sentiment_model_comparison_rv_mse.csv"
        comparison_df.to_csv(comparison_output, index=False)
        print(f"\n✅ Comparison results saved to: {comparison_output}")
        
        print("\n" + "="*60)
        print("COMPARISON COMPLETE")
        print("="*60)
        
else:
    print(f"\n⚠️  Forecast results file not found.")
    print(f"   Searched for:")
    for candidate in forecast_candidates:
        print(f"     - {candidate}")
    print(f"\n   Please run SVM2_VADER_Combined.py first to generate forecast results.")
    print(f"   Then re-run this script to perform the comparison.")
