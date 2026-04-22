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
    df = pd.read_csv(file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Step 1: Fix dtypes FIRST, then fill missing ──────────────────────────
    for col in df.columns:
        if df[col].dtype == "object":
            # Try converting to numeric; if mostly numeric, keep as numeric
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0.8 * len(df):
                df[col] = converted  # treat as numeric
            # else leave as object (categorical)

    # ── Step 2: Fill missing values AFTER dtype fix ───────────────────────────
    for col in df.columns:
        if df[col].dtype == "object":
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "missing")
        else:
            df[col] = df[col].fillna(df[col].median())

    # ── Step 3: Choose target column ─────────────────────────────────────────
    if "Risk_3Class" in df.columns:
        target_col = "Risk_3Class"
    elif "Risk" in df.columns:
        target_col = "Risk"
    else:
        st.error("❌ No target column found. Dataset must have 'Risk' or 'Risk_3Class' column.")
        st.stop()

    st.info(f"✅ Target column detected: **{target_col}**")

    # ── Step 4: Split features and target ─────────────────────────────────────
    drop_cols = [c for c in ["Risk", "Risk_3Class"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # ── Step 5: Encode target ─────────────────────────────────────────────────
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y.astype(str))

    # ── Step 6: Encode categorical features ───────────────────────────────────
    encoders = {}
    X_encoded = X.copy()
    for col in X_encoded.columns:
        if X_encoded[col].dtype == "object":
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le

    # ── Step 7: Final NaN check with imputer (safety net) ────────────────────
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

    st.success(f"✅ Model trained successfully! Accuracy: **{acc:.2%}**")

    # ── Step 9: Show classification report ───────────────────────────────────
    with st.expander("📋 View Detailed Classification Report"):
        report = classification_report(
            y_test, y_pred,
            target_names=target_le.classes_,
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(2))

    # ── Step 10: Prediction interface ─────────────────────────────────────────
    st.subheader("🔍 Make a Prediction")
    st.markdown("Fill in the borrower details below:")

    user_input = {}
    cols = st.columns(2)  # two-column layout for cleaner UI
    for i, col in enumerate(X.columns):
        with cols[i % 2]:
            if col in encoders:
                options = list(encoders[col].classes_)
                user_input[col] = st.selectbox(col, options)
            else:
                default_value = float(X[col].median()) if pd.api.types.is_numeric_dtype(X[col]) else 0.0
                user_input[col] = st.number_input(col, value=default_value)

    if st.button("🚀 Predict Risk"):
        input_df = pd.DataFrame([user_input])

        # Encode categorical inputs
        for col, le in encoders.items():
            val = input_df[col].astype(str)
            # Handle unseen labels gracefully
            known = set(le.classes_)
            input_df[col] = val.apply(lambda x: x if x in known else le.classes_[0])
            input_df[col] = le.transform(input_df[col])

        # Ensure column order matches training
        input_df = input_df[X_encoded.columns]
        input_imputed = imputer.transform(input_df)

        prediction = model.predict(input_imputed)[0]
        predicted_label = target_le.inverse_transform([prediction])[0]
        proba = model.predict_proba(input_imputed)[0]
        confidence = proba.max() * 100

        st.subheader("📌 Prediction Result")

        # Color-coded result
        color_map = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "good": "🟢", "bad": "🔴"}
        icon = color_map.get(predicted_label, "⚪")

        st.markdown(f"### {icon} Predicted Risk Category: **{predicted_label}**")
        st.markdown(f"**Confidence:** {confidence:.1f}%")

        # Probability breakdown
        st.markdown("**Probability Breakdown:**")
        proba_df = pd.DataFrame({
            "Risk Category": target_le.classes_,
            "Probability (%)": (proba * 100).round(1)
        })
        st.dataframe(proba_df)
