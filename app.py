# =============================================================================
#  LOAN REPAYMENT RISK PREDICTION — COMPLETE PIPELINE
#  Applied Business Analytics Project
#  Dataset: German Credit (Combined Version)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score
)

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================
df = pd.read_csv("german_credit_combined.csv", index_col=0)
print("=" * 60)
print("STEP 1: DATA LOADED")
print(f"  Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Columns: {df.columns.tolist()}")

# =============================================================================
# STEP 2 — DATA CLEANING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING")

# 2a. Missing value report
missing = df.isnull().sum()
missing = missing[missing > 0]
print("\n  Missing values:")
for col, count in missing.items():
    pct = count / len(df) * 100
    print(f"    {col:<25}: {count} ({pct:.1f}%)")

# 2b. Remove duplicate rows
dupes = df.duplicated().sum()
df = df.drop_duplicates()
print(f"\n  Duplicate rows removed: {dupes}")

# 2c. Drop columns not useful for prediction
# 'Personal Status' mixes gender+marital status (redundant with 'Sex')
# 'Other Debtors' has very low importance (0.022) — borderline, keep for now
COLS_TO_DROP = []  # We will drop weak features after importance analysis
print(f"  Columns dropped at this stage: {COLS_TO_DROP if COLS_TO_DROP else 'None (feature selection done later)'}")

# 2d. Impute missing values
#   'Saving accounts'  → 183 missing (18.3%) → fill with mode ('little')
#   'Checking account' → 394 missing (39.4%) → treat as own category 'unknown'
df['Saving accounts']  = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

print("\n  Missing values after imputation:")
print(f"    Saving accounts  : {df['Saving accounts'].isnull().sum()}")
print(f"    Checking account : {df['Checking account'].isnull().sum()}")
print("\n  ✅ Cleaning complete.")

# =============================================================================
# STEP 3 — ENCODE CATEGORICAL VARIABLES
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: ENCODING CATEGORICAL VARIABLES")

TARGET_COLS = ['Risk', 'Risk_3Class']
FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS]

# Separate feature matrix
X_raw = df[FEATURE_COLS].copy()
y_binary = df['Risk']           # good / bad
y_multiclass = df['Risk_3Class'] # Low / Medium / High

# Encode targets
binary_le = LabelEncoder()
y_bin_enc = binary_le.fit_transform(y_binary.astype(str))     # bad=0, good=1
print(f"  Binary classes  : {list(binary_le.classes_)}")

multi_le = LabelEncoder()
y_multi_enc = multi_le.fit_transform(y_multiclass.astype(str)) # High=0, Low=1, Medium=2
print(f"  Multiclass labels: {list(multi_le.classes_)}")

# Encode feature columns
encoders = {}
X_encoded = X_raw.copy()
cat_cols = [c for c in X_encoded.columns if X_encoded[c].dtype == object or str(X_encoded[c].dtype) == 'str']
num_cols = [c for c in X_encoded.columns if c not in cat_cols]

print(f"\n  Categorical features ({len(cat_cols)}): {cat_cols}")
print(f"  Numeric features  ({len(num_cols)}): {num_cols}")

for col in cat_cols:
    X_encoded[col] = X_encoded[col].astype(str)
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    encoders[col] = le

# Force all to numeric
X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')

# Final imputer safety net
imputer = SimpleImputer(strategy='median')
X_final = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)

print(f"\n  Final feature matrix shape: {X_final.shape}")
print("  ✅ Encoding complete. No missing values remain.")

# =============================================================================
# STEP 4 — FEATURE IMPORTANCE (Random Forest)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: FEATURE IMPORTANCE — RANDOM FOREST (Binary Target)")

rf_selector = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
rf_selector.fit(X_final, y_bin_enc)

fi_df = pd.DataFrame({
    'Feature'   : X_final.columns,
    'Importance': rf_selector.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n  Feature Importance Rankings:")
print(f"  {'Rank':<5} {'Feature':<25} {'Importance':>12}  {'Keep?'}")
print("  " + "-" * 55)
for i, row in fi_df.iterrows():
    keep = "✅ KEEP" if row['Importance'] >= 0.02 else "❌ DROP"
    print(f"  {fi_df.index.get_loc(i)+1:<5} {row['Feature']:<25} {row['Importance']:>12.4f}  {keep}")

# =============================================================================
# STEP 5 — REMOVE WEAK FEATURES (importance < 0.02)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: REMOVING WEAK FEATURES")

IMPORTANCE_THRESHOLD = 0.02
strong_features = fi_df[fi_df['Importance'] >= IMPORTANCE_THRESHOLD]['Feature'].tolist()
weak_features   = fi_df[fi_df['Importance'] <  IMPORTANCE_THRESHOLD]['Feature'].tolist()

print(f"  Threshold : {IMPORTANCE_THRESHOLD}")
print(f"  Removed   : {weak_features}")
print(f"  Kept      : {strong_features}")

X_selected = X_final[strong_features]
print(f"\n  Reduced feature matrix: {X_selected.shape[0]} rows × {X_selected.shape[1]} columns")

# =============================================================================
# STEP 6 — TRAIN / TEST SPLIT
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: TRAIN / TEST SPLIT (80/20, Stratified)")

X_train, X_test, y_train_bin, y_test_bin = train_test_split(
    X_selected, y_bin_enc,
    test_size=0.2, random_state=42, stratify=y_bin_enc
)
_, _, y_train_multi, y_test_multi = train_test_split(
    X_selected, y_multi_enc,
    test_size=0.2, random_state=42, stratify=y_multi_enc
)

print(f"  Train : {X_train.shape[0]} rows  |  Test : {X_test.shape[0]} rows")

# =============================================================================
# STEP 7 — MODEL A: LOGISTIC REGRESSION (Binary)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: MODEL A — LOGISTIC REGRESSION (Binary: good/bad)")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
lr.fit(X_train_sc, y_train_bin)
y_pred_lr = lr.predict(X_test_sc)

print(f"\n  Accuracy  : {accuracy_score(y_test_bin, y_pred_lr):.4f}")
print(f"  Precision : {precision_score(y_test_bin, y_pred_lr, average='weighted'):.4f}")
print(f"  Recall    : {recall_score(y_test_bin, y_pred_lr, average='weighted'):.4f}")
print(f"  F1-Score  : {f1_score(y_test_bin, y_pred_lr, average='weighted'):.4f}")
print("\n  Classification Report:")
print(classification_report(y_test_bin, y_pred_lr, target_names=binary_le.classes_))

# =============================================================================
# STEP 8 — MODEL B: RANDOM FOREST (Binary) — MAIN MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: MODEL B — RANDOM FOREST (Binary: good/bad) — MAIN MODEL")

rf_binary = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
rf_binary.fit(X_train, y_train_bin)
y_pred_rf = rf_binary.predict(X_test)

print(f"\n  Accuracy  : {accuracy_score(y_test_bin, y_pred_rf):.4f}")
print(f"  Precision : {precision_score(y_test_bin, y_pred_rf, average='weighted'):.4f}")
print(f"  Recall    : {recall_score(y_test_bin, y_pred_rf, average='weighted'):.4f}")
print(f"  F1-Score  : {f1_score(y_test_bin, y_pred_rf, average='weighted'):.4f}")
print("\n  Classification Report:")
print(classification_report(y_test_bin, y_pred_rf, target_names=binary_le.classes_))

# Cross-validation comparison
print("  Cross-Validation (5-fold F1 Weighted):")
lr_cv = cross_val_score(lr, X_train_sc, y_train_bin, cv=5, scoring='f1_weighted')
rf_cv = cross_val_score(rf_binary, X_train, y_train_bin, cv=5, scoring='f1_weighted')
print(f"    LR : {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
print(f"    RF : {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
winner = "Random Forest" if rf_cv.mean() > lr_cv.mean() else "Logistic Regression"
print(f"\n  ✅ Winner: {winner}")

# =============================================================================
# STEP 9 — MODEL C: RANDOM FOREST (3-Class) — EXTENDED MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 9: MODEL C — RANDOM FOREST (3-Class: Low/Medium/High)")

rf_multi = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
rf_multi.fit(X_train, y_train_multi)
y_pred_multi = rf_multi.predict(X_test)

print(f"\n  Accuracy  : {accuracy_score(y_test_multi, y_pred_multi):.4f}")
print(f"  Precision : {precision_score(y_test_multi, y_pred_multi, average='weighted'):.4f}")
print(f"  Recall    : {recall_score(y_test_multi, y_pred_multi, average='weighted'):.4f}")
print(f"  F1-Score  : {f1_score(y_test_multi, y_pred_multi, average='weighted'):.4f}")
print("\n  Classification Report:")
print(classification_report(y_test_multi, y_pred_multi, target_names=multi_le.classes_))

# =============================================================================
# STEP 10 — CONFUSION MATRICES & CHARTS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 10: SAVING EVALUATION CHARTS")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Loan Risk Prediction — Model Evaluation", fontsize=14, fontweight='bold')

# Confusion Matrix — LR Binary
cm_lr = confusion_matrix(y_test_bin, y_pred_lr)
ConfusionMatrixDisplay(cm_lr, display_labels=binary_le.classes_).plot(ax=axes[0], colorbar=False)
axes[0].set_title("Logistic Regression\n(Binary: good/bad)")

# Confusion Matrix — RF Binary
cm_rf = confusion_matrix(y_test_bin, y_pred_rf)
ConfusionMatrixDisplay(cm_rf, display_labels=binary_le.classes_).plot(ax=axes[1], colorbar=False)
axes[1].set_title("Random Forest\n(Binary: good/bad)")

# Confusion Matrix — RF 3-Class
cm_multi = confusion_matrix(y_test_multi, y_pred_multi)
ConfusionMatrixDisplay(cm_multi, display_labels=multi_le.classes_).plot(ax=axes[2], colorbar=False)
axes[2].set_title("Random Forest\n(3-Class: Low/Med/High)")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: confusion_matrices.png")

# Feature importance chart
fig2, ax2 = plt.subplots(figsize=(10, 6))
colors = ['#2ecc71' if imp >= 0.02 else '#e74c3c' for imp in fi_df['Importance']]
bars = ax2.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1], color=colors[::-1])
ax2.axvline(x=0.02, color='black', linestyle='--', linewidth=1, label='Threshold (0.02)')
ax2.set_xlabel("Feature Importance Score")
ax2.set_title("Feature Importance — Random Forest (Binary Target)")
ax2.legend()
for bar, val in zip(bars, fi_df['Importance'][::-1]):
    ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: feature_importance.png")

# Model comparison bar chart
fig3, ax3 = plt.subplots(figsize=(8, 5))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
lr_vals = [
    accuracy_score(y_test_bin, y_pred_lr),
    precision_score(y_test_bin, y_pred_lr, average='weighted'),
    recall_score(y_test_bin, y_pred_lr, average='weighted'),
    f1_score(y_test_bin, y_pred_lr, average='weighted')
]
rf_vals = [
    accuracy_score(y_test_bin, y_pred_rf),
    precision_score(y_test_bin, y_pred_rf, average='weighted'),
    recall_score(y_test_bin, y_pred_rf, average='weighted'),
    f1_score(y_test_bin, y_pred_rf, average='weighted')
]
x = np.arange(len(metrics))
w = 0.35
ax3.bar(x - w/2, lr_vals, w, label='Logistic Regression', color='#3498db')
ax3.bar(x + w/2, rf_vals, w, label='Random Forest',       color='#2ecc71')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics)
ax3.set_ylim(0, 1.0)
ax3.set_ylabel("Score")
ax3.set_title("LR vs Random Forest — Performance Comparison")
ax3.legend()
for i, (lv, rv) in enumerate(zip(lr_vals, rf_vals)):
    ax3.text(i - w/2, lv + 0.01, f'{lv:.2f}', ha='center', fontsize=9)
    ax3.text(i + w/2, rv + 0.01, f'{rv:.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: model_comparison.png")

print("\n" + "=" * 60)
print("✅ PIPELINE COMPLETE — All steps done.")
print("   Files saved: confusion_matrices.png, feature_importance.png, model_comparison.png")
