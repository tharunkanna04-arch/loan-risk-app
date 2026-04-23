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

st.set_page_config(page_title="Loan Risk Predictor", layout="wide")
st.title("🏦 Loan Repayment Risk Predictor")

# ── Helper ────────────────────────────────────────────────────────────────────
def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", make_onehot())]),    cat_cols),
    ], remainder="drop")
    return Pipeline([
        ("pre",   pre),
        ("model", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")),
    ])

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload any CSV dataset", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file).replace(r"^\s*$", np.nan, regex=True)

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)
st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Target picker ─────────────────────────────────────────────────────────────
st.divider()
common = ["Risk", "Risk_3Class", "loan_status", "default", "target", "label", "outcome"]
default_target = next((c for c in common if c in df.columns), df.columns[-1])

target_col = st.selectbox("Select target column", df.columns.tolist(),
                           index=df.columns.tolist().index(default_target))

exclude_cols = st.multiselect("Exclude columns (IDs, names, irrelevant)",
                               [c for c in df.columns if c != target_col])

# ── Prepare features ──────────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c != target_col and c not in exclude_cols]
X = df[feature_cols].copy()
y_raw = df[target_col].astype(str).str.strip()

if y_raw.nunique() < 2:
    st.error("Target has only one class. Choose a different column.")
    st.stop()

# Drop constant columns
const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
if const:
    X = X.drop(columns=const)
    st.info(f"Removed constant columns: {const}")

y_le = LabelEncoder()
y    = y_le.fit_transform(y_raw)
classes = y_le.classes_

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
st.success(f"Target: **{target_col}** | Classes: {list(classes)} | Features: {len(X.columns)}")

# ── Train ─────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Model Results")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

clf = build_pipeline(num_cols, cat_cols)

with st.spinner("Training..."):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Baseline
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_acc  = accuracy_score(y_test, dummy_pred)

    # Cross-validation
    cv_scores = cross_val_score(
        build_pipeline(num_cols, cat_cols), X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_weighted"
    )

# ── Metrics ───────────────────────────────────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy",       f"{acc:.2%}", delta=f"{acc - dummy_acc:+.2%} vs baseline")
c2.metric("Precision",      f"{prec:.2%}")
c3.metric("Recall",         f"{rec:.2%}")
c4.metric("F1 Score",       f"{f1:.2%}")
c5.metric("CV F1 (5-fold)", f"{cv_scores.mean():.2%}")
c6.metric("CV Std",         f"±{cv_scores.std():.2%}")

gap = acc - dummy_acc
if gap < 0.03:
    st.error("⚠️ Model barely beats the baseline — features may not be predictive enough.")
elif gap < 0.08:
    st.warning("⚠️ Model is only slightly better than baseline.")
else:
    st.success(f"✅ Model beats the always-predict-majority baseline by {gap:.1%}.")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
    index  =[f"Actual: {c}"    for c in classes],
    columns=[f"Predicted: {c}" for c in classes])
st.dataframe(cm_df, use_container_width=True)

# ── Classification Report ─────────────────────────────────────────────────────
with st.expander("Full Classification Report"):
    report = classification_report(y_test, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

# ── Feature Importance ────────────────────────────────────────────────────────
st.subheader("Top Influential Variables")
try:
    feat_names = clf.named_steps["pre"].get_feature_names_out()
    coefs      = clf.named_steps["model"].coef_
    importance = np.abs(coefs.ravel()) if coefs.shape[0] == 1 else np.mean(np.abs(coefs), axis=0)
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importance}) \
              .sort_values("Importance", ascending=False).head(10).reset_index(drop=True)
    fi_df.index += 1
    st.dataframe(fi_df, use_container_width=True)
    st.caption("Higher value = stronger influence on prediction.")
except Exception as e:
    st.warning(f"Could not extract feature importances: {e}")

# ── Prediction Form ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Make a Prediction")

user_input = {}
form_cols  = st.columns(3)
for idx, col in enumerate(X.columns):
    with form_cols[idx % 3]:
        if col in num_cols:
            med = float(pd.to_numeric(X[col], errors="coerce").median())
            user_input[col] = st.number_input(col, value=med if not np.isnan(med) else 0.0)
        else:
            opts = sorted(X[col].astype(str).fillna("missing").unique().tolist())
            user_input[col] = st.selectbox(col, opts)

if st.button("Predict", type="primary"):
    input_df    = pd.DataFrame([user_input])
    pred_idx    = clf.predict(input_df)[0]
    pred_label  = y_le.inverse_transform([pred_idx])[0]
    proba       = clf.predict_proba(input_df)[0]
    confidence  = proba[pred_idx] * 100

    risk_words = {"bad", "high", "default", "1", "yes", "risky"}
    if str(pred_label).lower() in risk_words:
        st.error(f"⚠️ Predicted: **{pred_label.upper()}**  |  Confidence: {confidence:.1f}%")
    else:
        st.success(f"✅ Predicted: **{pred_label.upper()}**  |  Confidence: {confidence:.1f}%")

    st.dataframe(pd.DataFrame({"Class": classes,
                                "Probability (%)": (proba * 100).round(2)}),
                 use_container_width=True)
    
