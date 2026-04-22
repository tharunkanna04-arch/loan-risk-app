import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

st.title("🏦 Loan Repayment Risk Predictor")
st.markdown("Upload your dataset to train the model and make predictions.")

file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if file is not None:
    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(file)

    # Force all columns to simple native Python dtypes (fixes PyArrow string issue)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce") if pd.api.types.is_numeric_dtype(df[col]) else df[col].astype(str)
        except Exception:
            df[col] = df[col].astype(str)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Step 1: Fill missing values ───────────────────────────────────────────
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "missing")

    # ── Step 2: Choose target column ─────────────────────────────────────────
    if "Risk_3Class" in df.columns:
        target_col = "Risk_3Class"
    elif "Risk" in df.columns:
        target_col = "Risk"
    else:
        st.error("❌ No target column found. Dataset must have 'Risk' or 'Risk_3Class' column.")
        st.stop()

    st.info(f"✅ Target column detected: **{target_col}**")

    # ── Step 3: Split features and target ─────────────────────────────────────
    drop_cols = [c for c in ["Risk", "Risk_3Class"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # ── Step 4: Encode target ─────────────────────────────────────────────────
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y.astype(str))

    # ── Step 5: Encode ALL non-numeric columns (bulletproof) ──────────────────
    encoders = {}
    X_encoded = X.copy()
    for col in X_encoded.columns:
        if not pd.api.types.is_numeric_dtype(X_encoded[col]):
            X_encoded[col] = X_encoded[col].astype(str)
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
            encoders[col] = le

    # ── Step 6: Force everything to float64 ───────────────────────────────────
    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")

    # ── Step 7: Final imputer as safety net ───────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_encoded)
    X_imputed = pd.DataFrame(X_imputed, columns=X_encoded.columns)

    # ── Step 8: Train model ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ Model trained! Accuracy: **{acc:.2%}**")

    # ── Step 9: Classification report ─────────────────────────────────────────
    with st.expander("📋 View Detailed Classification Report"):
        report = classification_report(
            y_test, y_pred,
            target_names=target_le.classes_,
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(2))

    # ── Step 10: Prediction UI ────────────────────────────────────────────────
    st.subheader("🔍 Make a Prediction")
    st.markdown("Fill in the borrower details below:")

    user_input = {}
    ui_cols = st.columns(2)
    for i, col in enumerate(X.columns):
        with ui_cols[i % 2]:
            if col in encoders:
                options = list(encoders[col].classes_)
                user_input[col] = st.selectbox(col, options)
            else:
                default_val = float(X[col].median()) if pd.api.types.is_numeric_dtype(X[col]) else 0.0
                user_input[col] = st.number_input(col, value=default_val)

    if st.button("🚀 Predict Risk"):
        input_df = pd.DataFrame([user_input])

        for col, le in encoders.items():
            val = str(input_df[col].iloc[0])
            known = set(le.classes_)
            val = val if val in known else le.classes_[0]
            input_df[col] = le.transform([val])

        input_df = input_df[X_encoded.columns]
        input_df = input_df.apply(pd.to_numeric, errors="coerce")
        input_imputed = imputer.transform(input_df)

        prediction = model.predict(input_imputed)[0]
        predicted_label = target_le.inverse_transform([prediction])[0]
        proba = model.predict_proba(input_imputed)[0]
        confidence = proba.max() * 100

        st.subheader("📌 Prediction Result")
        color_map = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "good": "🟢", "bad": "🔴"}
        icon = color_map.get(predicted_label, "⚪")

        st.markdown(f"### {icon} Predicted Risk: **{predicted_label}**")
        st.markdown(f"**Confidence:** {confidence:.1f}%")

        st.markdown("**Probability Breakdown:**")
        proba_df = pd.DataFrame({
            "Risk Category": target_le.classes_,
            "Probability (%)": (proba * 100).round(1)
        })
        st.dataframe(proba_df)
