import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
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
st.write("Upload your CSV file. The app will clean the data, choose a valid target, and train Logistic Regression.")

uploaded_file = st.file_uploader("Upload dataset", type=["csv"])


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def choose_target(df: pd.DataFrame):
    for col in ["Risk", "Risk_3Class"]:
        if col in df.columns:
            vals = df[col].dropna().astype(str).str.strip()
            if vals.nunique() >= 2:
                return col
    return None


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.replace(r"^\s*$", np.nan, regex=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    target_col = choose_target(df)

    if target_col is None:
        st.error("No valid target found. The file must contain either 'Risk' or 'Risk_3Class' with at least 2 classes.")
        st.stop()

    st.success(f"Using target column: {target_col}")

    y_raw = df[target_col].astype(str).str.strip().str.lower()

    if y_raw.nunique() < 2:
        st.error(f"Target column '{target_col}' has only one class. Upload the raw dataset where the target has at least 2 classes.")
        st.stop()

    drop_cols = [c for c in ["Risk", "Risk_3Class"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    if X.shape[1] == 0:
        st.error("No feature columns left after removing target columns.")
        st.stop()

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
        st.info(f"Dropped constant columns: {constant_cols}")

    y_le = LabelEncoder()
    y = y_le.fit_transform(y_raw)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    st.write(f"Numeric features: {len(numeric_features)}")
    st.write(f"Categorical features: {len(categorical_features)}")

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot()),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.2%}")
    c2.metric("Precision", f"{prec:.2%}")
    c3.metric("Recall", f"{rec:.2%}")
    c4.metric("F1 Score", f"{f1:.2%}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=[f"Actual {x}" for x in y_le.classes_], columns=[f"Predicted {x}" for x in y_le.classes_])
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=y_le.classes_, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Most Influential Variables")
    feature_names = clf.named_steps["preprocess"].get_feature_names_out()
    model = clf.named_steps["model"]

    coefs = model.coef_
    if coefs.ndim == 1 or coefs.shape[0] == 1:
        importance = np.abs(coefs.ravel())
    else:
        importance = np.mean(np.abs(coefs), axis=0)

    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance,
        }
    ).sort_values("Importance", ascending=False)

    st.dataframe(coef_df.head(15), use_container_width=True)

    st.subheader("Make a Prediction")
    st.write("Fill the borrower details below:")

    user_input = {}
    form_cols = st.columns(2)

    for idx, col in enumerate(X.columns):
        with form_cols[idx % 2]:
            if col in numeric_features:
                default_val = float(pd.to_numeric(X[col], errors="coerce").median())
                if pd.isna(default_val):
                    default_val = 0.0
                user_input[col] = st.number_input(col, value=default_val)
            else:
                options = sorted(X[col].astype(str).fillna("missing").unique().tolist())
                user_input[col] = st.selectbox(col, options)

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        pred = clf.predict(input_df)[0]
        pred_label = y_le.inverse_transform([pred])[0]

        st.success(f"Predicted class: {pred_label}")

        try:
            proba = clf.predict_proba(input_df)[0]
            proba_df = pd.DataFrame(
                {
                    "Class": y_le.classes_,
                    "Probability": (proba * 100).round(2),
                }
            )
            st.dataframe(proba_df, use_container_width=True)
        except Exception:
            pass
else:
    st.info("Upload a CSV file to start.")
