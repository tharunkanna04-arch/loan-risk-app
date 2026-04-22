import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.title("Loan Repayment Risk Predictor")

file = st.file_uploader("Upload your dataset", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    df = df.fillna(0)
    
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    
    X = df.drop(['Risk', 'Risk_3Class'], axis=1)
    y = df['Risk']
    
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)
    
    st.write("Enter values:")
    
    input_data = []
    for col in X.columns:
        val = st.number_input(col, value=0.0)
        input_data.append(val)
    
    if st.button("Predict"):
        result = model.predict([input_data])
        st.success(f"Prediction: {result[0]}")
