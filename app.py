import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="Loan Repayment Risk Predictor", layout="wide")
st.title("Loan Repayment Risk Predictor")
st.write("Upload your CSV file. The app will keep only statistically significant variables and train Logistic Regression.")

uploaded_file = st.file_uploader("Upload dataset", type=["csv"])


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace("nan", np.nan)
            mode_vals = df[col].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else "missing"
            df[col] = df[col].fillna(fill_value)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)

    return df


def encode_binary_target(y: pd.Series) -> pd.Series:
    y_str = y.astype(str).str.lower().str.strip()

    mapping = {
        "bad": 0,
        "good": 1,
        "0": 0,
        "1": 1,
        "no": 0,
        "yes": 1,
        "default": 0,
        "non-default": 1,
    }

    mapped = y_str.map(mapping)
    if mapped.isna().any():
        unique_vals = sorted(y_str.dropna().unique())
        if len(unique_vals) != 2:
            raise ValueError(
                f"Target column must be binary. Found values: {unique_vals}"
            )
        # fallback: first label = 0, second label = 1
        fallback_map = {unique_vals[0]: 0, unique_vals[1]: 1}
        mapped = y_str.map(fallback_map)

    return mapped.astype(int)


def get_significant_features(X: pd.DataFrame, y: pd.Series, alpha: float = 0.05):
    X_dum = pd.get_dummies(X, drop_first=True)
    selected = list(X_dum.columns)

    last_result = None

    while True:
        X_sm = sm.add_constant(X_dum[selected], has_constant="add")

        try:
            model = sm.GLM(y, X_sm, family=sm.families.Binomial())
            result = model.fit()
            last_result = result
        except Exception as e:
            st.warning(f"Statsmodels could not fit perfectly. Stopping feature removal here. Details: {e}")
            break

        pvals = result.pvalues.drop("const", errors="ignore")
        if pvals.empty:
            break

        max_p = pvals.max()
        if max_p <= alpha or len(selected) <= 1:
            break

        worst_feature = pvals.idxmax()
        selected.remove(worst_feature)

    if last_result is None:
        raise ValueError("Could not fit a statistical model for p-value selection.")

    final_pvals = last_result.pvalues.drop("const", errors="ignore").sort_values()
    return X_dum, selected, final_pvals, last_result


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "Risk" not in df.columns:
        st.error("This dataset must contain a binary target column named 'Risk'.")
        st.stop()

    df = clean_dataframe(df)

    # binary target only
    y = encode_binary_target(df["Risk"])

    drop_cols = [c for c in ["Risk", "Risk_3Class"] if c in df.columns]
    X_raw = df.drop(columns=drop_cols)

    st.subheader("Step 1: Build statistically significant feature set")
    try:
        X_encoded, significant_features, pvals_table, stat_result = get_significant_features(X_raw, y, alpha=0.05)
    except Exception as e:
        st.error(f"Feature selection failed: {e}")
        st.stop()

    st.write(f"Original feature count after one-hot encoding: {X_encoded.shape[1]}")
    st.write(f"Selected significant features: {len(significant_features)}")

    pval_df = pd.DataFrame({
        "Variable": pvals_table.index,
        "P-value": pvals_table.values
    })
    st.dataframe(pval_df, use_container_width=True)

    if len(significant_features) == 0:
        st.error("No significant variables were found.")
        st.stop()

    X_selected = X_encoded[significant_features]

    st.subheader("Step 2: Train Logistic Regression on significant variables")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.2%}")
    c2.metric("Precision", f"{prec:.2%}")
    c3.metric("Recall", f"{rec:.2%}")
    c4.metric("F1 Score", f"{f1:.2%}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Bad", "Actual Good"],
        columns=["Predicted Bad", "Predicted Good"]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Predict a New Borrower")

    user_input = {}
    col1, col2 = st.columns(2)
    input_cols = list(X_raw.columns)

    for idx, col in enumerate(input_cols):
        box = col1 if idx % 2 == 0 else col2

        with box:
            if pd.api.types.is_numeric_dtype(X_raw[col]):
                default_val = float(pd.to_numeric(X_raw[col], errors="coerce").median())
                if pd.isna(default_val):
                    default_val = 0.0
                user_input[col] = st.number_input(col, value=default_val)
            else:
                options = sorted(X_raw[col].astype(str).unique().tolist())
                user_input[col] = st.selectbox(col, options)

    if st.button("Predict Risk"):
        new_row = pd.DataFrame([user_input])
        new_row_encoded = pd.get_dummies(new_row, drop_first=True)

        # align with training columns
        new_row_encoded = new_row_encoded.reindex(columns=X_encoded.columns, fill_value=0)
        new_row_selected = new_row_encoded[significant_features]
        new_row_scaled = scaler.transform(new_row_selected)

        pred = model.predict(new_row_scaled)[0]
        prob = model.predict_proba(new_row_scaled)[0]

        label = "Good" if pred == 1 else "Bad"
        st.success(f"Predicted Risk: {label}")
        st.write(f"Probability of Bad: {prob[0]:.2%}")
        st.write(f"Probability of Good: {prob[1]:.2%}")
else:
    st.info("Upload a CSV file to start.")
