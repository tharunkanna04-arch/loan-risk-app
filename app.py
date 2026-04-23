import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Universal Risk Predictor", layout="wide")
st.title("🏦 Universal Classification Predictor")
st.caption("Works with any CSV dataset — any domain, any target column.")

# ── Helper: OneHotEncoder compatibility ──────────────────────────────────────
def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# ── Helper: build sklearn pipeline ───────────────────────────────────────────
def build_pipeline(numeric_features, categorical_features, C=1.0):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  make_onehot()),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe,      numeric_features),
        ("cat", categorical_pipe,  categorical_features),
    ], remainder="drop")
    clf = Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(
            max_iter=5000, class_weight="balanced",
            solver="lbfgs", C=C
        )),
    ])
    return clf

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader("📂 Upload any CSV dataset", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin. Works with any classification dataset.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)
df = df_raw.replace(r"^\s*$", np.nan, regex=True).copy()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)
st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COLUMN CONFIGURATION (generic — user picks everything)
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("⚙️ Step 1 — Configure Columns")

col_left, col_right = st.columns(2)

with col_left:
    # Target column — user picks from dropdown (no hardcoding)
    all_cols = df.columns.tolist()
    
    # Smart default: try common target names first
    common_targets = ["Risk", "Risk_3Class", "loan_status", "default", "target",
                      "label", "class", "outcome", "y", "Default"]
    default_target = next((c for c in common_targets if c in all_cols), all_cols[-1])
    default_idx = all_cols.index(default_target)
    
    target_col = st.selectbox(
        "🎯 Select Target (dependent) column",
        options=all_cols,
        index=default_idx,
        help="The column you want to predict."
    )

with col_right:
    # Columns to exclude (ID, names, irrelevant cols)
    exclude_cols = st.multiselect(
        "🗑️ Exclude columns (IDs, names, irrelevant)",
        options=[c for c in all_cols if c != target_col],
        help="Remove ID numbers, row numbers, names — anything not useful for prediction."
    )

# Build feature set
feature_cols = [c for c in all_cols if c != target_col and c not in exclude_cols]

if len(feature_cols) == 0:
    st.error("No feature columns left. Please adjust your exclusions.")
    st.stop()

X = df[feature_cols].copy()
y_raw = df[target_col].astype(str).str.strip()

# Validate target
if y_raw.nunique() < 2:
    st.error(f"'{target_col}' has only one unique value. Choose a different target column.")
    st.stop()

if y_raw.nunique() > 20:
    st.warning(f"⚠️ '{target_col}' has {y_raw.nunique()} unique values. "
               "This tool is for classification (not regression). "
               "Consider choosing a column with fewer categories.")

# Drop constant feature columns
constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
if constant_cols:
    X = X.drop(columns=constant_cols)
    st.info(f"Auto-removed constant columns: {constant_cols}")

# Encode target
y_le = LabelEncoder()
y = y_le.fit_transform(y_raw)
classes = y_le.classes_

st.success(f"✅ Target: **{target_col}** | Classes: {list(classes)} | Features: {len(X.columns)}")

# Show feature breakdown
numeric_features     = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

fc1, fc2, fc3 = st.columns(3)
fc1.metric("Total Features",      len(X.columns))
fc2.metric("Numeric Features",    len(numeric_features))
fc3.metric("Categorical Features",len(categorical_features))

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("⚙️ Step 2 — Model Settings")

s1, s2, s3 = st.columns(3)
with s1:
    C_value = st.select_slider(
        "Regularisation strength (C)",
        options=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        value=1.0,
        help="Higher C = less regularisation = model fits more closely to training data."
    )
with s2:
    test_size = st.slider("Test set size (%)", min_value=10, max_value=40, value=20, step=5)
with s3:
    threshold = st.slider(
        "Decision threshold",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Lower = catch more positive cases (e.g. risky borrowers). Default = 0.50."
    )

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAIN MODEL
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("🚀 Step 3 — Train & Evaluate")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100,
    random_state=42, stratify=y
)

clf = build_pipeline(numeric_features, categorical_features, C=C_value)

with st.spinner("Training model..."):
    clf.fit(X_train, y_train)

# Apply custom threshold
y_proba = clf.predict_proba(X_test)

if len(classes) == 2:
    # Binary: apply threshold to positive class (index 1)
    y_pred = (y_proba[:, 1] >= threshold).astype(int)
else:
    # Multiclass: argmax (threshold slider has no effect)
    y_pred = np.argmax(y_proba, axis=1)
    if threshold != 0.50:
        st.info("ℹ️ Decision threshold only applies to binary classification.")

# ── Core metrics ─────────────────────────────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

# ── Baseline: dummy classifier ────────────────────────────────────────────────
dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_acc  = accuracy_score(y_test, dummy_pred)
dummy_f1   = f1_score(y_test, dummy_pred, average="weighted", zero_division=0)

# ── Cross-validation ──────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
with st.spinner("Running 5-fold cross-validation..."):
    cv_scores = cross_val_score(
        build_pipeline(numeric_features, categorical_features, C=C_value),
        X, y, cv=cv, scoring="f1_weighted"
    )

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RESULTS
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("📈 Model Performance")

# Row 1: main metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy",        f"{acc:.2%}", delta=f"{(acc - dummy_acc):+.2%} vs baseline")
m2.metric("Precision",       f"{prec:.2%}")
m3.metric("Recall",          f"{rec:.2%}")
m4.metric("F1 (Weighted)",   f"{f1:.2%}")
m5.metric("F1 (Macro)",      f"{f1_macro:.2%}")

# Row 2: validation metrics
v1, v2, v3, v4 = st.columns(4)
v1.metric("CV F1 Mean (5-fold)", f"{cv_scores.mean():.2%}")
v2.metric("CV F1 Std",           f"± {cv_scores.std():.2%}",
          help="Low std = stable model. High std = unstable — needs more data.")
v3.metric("Baseline Accuracy",   f"{dummy_acc:.2%}",
          help="A dumb model that always predicts the majority class.")
v4.metric("Baseline F1",         f"{dummy_f1:.2%}")

# Reliability warning
gap = acc - dummy_acc
if gap < 0.03:
    st.error("⚠️ Your model barely beats the baseline. The features may not be predictive enough.")
elif gap < 0.08:
    st.warning("⚠️ Model is only slightly better than baseline. Consider better features or more data.")
else:
    st.success(f"✅ Model beats baseline by {gap:.1%} — learning real patterns.")

if cv_scores.std() > 0.08:
    st.warning("⚠️ High CV variance (±{:.1%}) — model is unstable. Consider collecting more data.".format(cv_scores.std()))

# Decision threshold note
if len(classes) == 2 and threshold != 0.50:
    st.info(f"ℹ️ Using decision threshold = {threshold:.2f} (default 0.50). "
            f"This shifts the trade-off between catching positives vs false alarms.")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
st.subheader("🔢 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index   =[f"Actual → {c}"    for c in classes],
    columns =[f"Predicted → {c}" for c in classes]
)
st.dataframe(cm_df, use_container_width=True)

if len(classes) == 2:
    tn, fp, fn, tp = cm.ravel()
    ci1, ci2, ci3, ci4 = st.columns(4)
    ci1.metric("True Negatives",  tn, help="Correctly predicted negative class")
    ci2.metric("False Positives", fp, help="Negative predicted as positive (Type I Error)")
    ci3.metric("False Negatives", fn, help="Positive predicted as negative (Type II Error — riskiest for banks)")
    ci4.metric("True Positives",  tp, help="Correctly predicted positive class")

# ── Classification Report ─────────────────────────────────────────────────────
st.subheader("📋 Classification Report")
report = classification_report(
    y_test, y_pred, target_names=classes,
    output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report).transpose().round(3)
st.dataframe(report_df, use_container_width=True)

# ── Feature Importance ────────────────────────────────────────────────────────
st.subheader("📊 Most Influential Variables")
try:
    feature_names = clf.named_steps["preprocess"].get_feature_names_out()
    coefs = clf.named_steps["model"].coef_
    importance = np.abs(coefs.ravel()) if coefs.shape[0] == 1 else np.mean(np.abs(coefs), axis=0)

    coef_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importance,
    }).sort_values("Importance", ascending=False).head(15).reset_index(drop=True)
    coef_df.index += 1
    st.dataframe(coef_df, use_container_width=True)
    st.caption("Importance = absolute value of logistic regression coefficient (after scaling). Higher = stronger predictor.")
except Exception as e:
    st.warning(f"Could not extract feature importances: {e}")

# ── Cross-validation detail ───────────────────────────────────────────────────
with st.expander("🔍 Cross-Validation Detail (5 folds)"):
    cv_df = pd.DataFrame({
        "Fold":     [f"Fold {i+1}" for i in range(5)],
        "F1 Score": [f"{s:.4f}" for s in cv_scores],
    })
    cv_df.loc[len(cv_df)] = ["Mean ± Std", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"]
    st.dataframe(cv_df, use_container_width=True)
    st.caption(
        "Cross-validation splits your data 5 ways and trains/tests each time. "
        "The mean F1 is a more reliable accuracy estimate than a single split."
    )

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PREDICTION FORM (generic — uses whatever columns are in the data)
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("🔍 Make a Prediction")
st.write("Fill in the values below for a new borrower / record:")

user_input = {}
form_cols = st.columns(3)

for idx, col in enumerate(X.columns):
    with form_cols[idx % 3]:
        if col in numeric_features:
            median_val = pd.to_numeric(X[col], errors="coerce").median()
            default_val = float(median_val) if not pd.isna(median_val) else 0.0
            user_input[col] = st.number_input(col, value=default_val, key=f"input_{col}")
        else:
            options = sorted(X[col].astype(str).fillna("missing").unique().tolist())
            user_input[col] = st.selectbox(col, options, key=f"input_{col}")

if st.button("🚀 Predict", type="primary"):
    input_df = pd.DataFrame([user_input])

    pred_proba = clf.predict_proba(input_df)[0]

    if len(classes) == 2:
        pred_class_idx = int(pred_proba[1] >= threshold)
    else:
        pred_class_idx = int(np.argmax(pred_proba))

    pred_label    = y_le.inverse_transform([pred_class_idx])[0]
    pred_confidence = pred_proba[pred_class_idx] * 100

    # Colour-coded result
    risk_keywords = {"bad", "high", "default", "1", "yes", "risky"}
    is_high_risk  = str(pred_label).lower() in risk_keywords

    if is_high_risk:
        st.error(f"⚠️ Predicted Class: **{pred_label.upper()}**  |  Confidence: {pred_confidence:.1f}%")
    else:
        st.success(f"✅ Predicted Class: **{pred_label.upper()}**  |  Confidence: {pred_confidence:.1f}%")

    proba_df = pd.DataFrame({
        "Class":          classes,
        "Probability (%)": (pred_proba * 100).round(2),
    })
    st.dataframe(proba_df, use_container_width=True)

    st.caption(f"Decision threshold used: {threshold:.2f}")
