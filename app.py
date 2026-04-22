import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("Loan Repayment Risk Predictor")

file = st.file_uploader("Upload your dataset", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # Clean missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "missing")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Choose target
    if "Risk_3Class" in df.columns:
        target_col = "Risk_3Class"
    elif "Risk" in df.columns:
        target_col = "Risk"
    else:
        st.error("No target column found. Please upload a dataset with 'Risk' or 'Risk_3Class'.")
        st.stop()

    # Features
    drop_cols = [c for c in ["Risk", "Risk_3Class"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Encode target
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y.astype(str))

    # Encode categorical feature columns
    encoders = {}
    X_encoded = X.copy()

    for col in X.columns:
        if X_encoded[col].dtype == "object":
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Model trained successfully. Accuracy: {acc:.2f}")

    st.subheader("Make a prediction")

    user_input = {}

    for col in X.columns:
        if col in encoders:
            options = list(encoders[col].classes_)
            user_input[col] = st.selectbox(col, options)
        else:
            default_value = float(X[col].median()) if pd.api.types.is_numeric_dtype(X[col]) else 0.0
            user_input[col] = st.number_input(col, value=default_value)

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])

        for col, le in encoders.items():
            input_df[col] = le.transform(input_df[col].astype(str))

        input_df = input_df[X.columns]
        prediction = model.predict(input_df)[0]
        predicted_label = target_le.inverse_transform([prediction])[0]

        st.subheader("Prediction Result")
        st.write(f"Predicted risk category: **{predicted_label}**")
