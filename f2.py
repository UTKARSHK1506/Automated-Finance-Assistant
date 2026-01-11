import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import joblib
import time
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from groq import Groq

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="💰 Smart Finance Assistant", layout="wide")
st.title("💰 Smart Finance Assistant")
st.caption("Upload bank CSV → View expenses → Predict savings → AI advice")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("finance_lr_model.pkl")
    feature_cols = joblib.load("finance_features.pkl")
    return model, feature_cols

model, FEATURE_COLS = load_model()

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
variable_expenses = [
    "Groceries", "Transport", "Eating_Out", "Entertainment",
    "Utilities", "Healthcare", "Education", "Miscellaneous"
]

numerical_features = ["Income", "Age", "Dependents"] + variable_expenses
categorical_features = ["Occupation", "City_Tier"]

# --------------------------------------------------
# CATEGORY LOGIC
# --------------------------------------------------
def categorize(desc):
    desc = str(desc).lower()
    if any(k in desc for k in ["salary", "credited", "income", "payroll"]):
        return "INCOME"
    if any(k in desc for k in ["grocery", "supermarket", "mart"]):
        return "Groceries"
    if any(k in desc for k in ["uber", "ola", "fuel", "bus", "train"]):
        return "Transport"
    if any(k in desc for k in ["food", "restaurant", "cafe", "zomato", "swiggy"]):
        return "Eating_Out"
    if any(k in desc for k in ["movie", "netflix", "spotify"]):
        return "Entertainment"
    if any(k in desc for k in ["electricity", "water", "gas", "bill"]):
        return "Utilities"
    if any(k in desc for k in ["hospital", "pharmacy", "medical"]):
        return "Healthcare"
    if any(k in desc for k in ["school", "college", "fees", "course"]):
        return "Education"
    return "Miscellaneous"

# --------------------------------------------------
# FEATURE BUILDER
# --------------------------------------------------
def build_features(df, amount_col, desc_col, type_col):
    features = {}

    df[type_col] = df[type_col].str.lower().str.strip()
    income_df = df[df[type_col].isin(["credit", "cr"])]
    expense_df = df[df[type_col].isin(["debit", "dr"])]

    features["Income"] = income_df[amount_col].sum()

    expense_df = expense_df.copy()
    expense_df["Category"] = expense_df[desc_col].apply(categorize)
    expense_df = expense_df[expense_df["Category"] != "INCOME"]

    for cat in variable_expenses:
        features[cat] = expense_df[expense_df["Category"] == cat][amount_col].sum()

    features["Age"] = 30
    features["Dependents"] = 0
    features["Occupation"] = "Private"
    features["City_Tier"] = 2

    X = pd.DataFrame([features])

    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded = encoder.fit_transform(X[categorical_features])
    enc_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_features))

    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    X = pd.concat([X.drop(columns=categorical_features), enc_df], axis=1)

    for col in FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0

    return X[FEATURE_COLS], expense_df

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload Bank CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    amount_col = next(c for c in df.columns if "amount" in c.lower())
    desc_col = next(c for c in df.columns if "desc" in c.lower() or "narration" in c.lower())
    type_col = next(c for c in df.columns if "type" in c.lower())

    df[amount_col] = df[amount_col].astype(float)

    X, expense_df = build_features(df, amount_col, desc_col, type_col)

    # --------------------------------------------------
    # EXPENSE CHART
    # --------------------------------------------------
    st.subheader("📊 Expense Breakdown")

    expense_plot = (
        expense_df.groupby("Category")[amount_col]
        .sum()
        .reset_index()
    )

    chart = alt.Chart(expense_plot).mark_bar().encode(
        x="Category",
        y=amount_col,
        color="Category"
    )
    st.altair_chart(chart, use_container_width=True)

    # --------------------------------------------------
    # SCALABLE SAVINGS LOGIC
    # --------------------------------------------------
    total_expense = expense_plot[amount_col].sum()
    savings = {}

    for _, row in expense_plot.iterrows():
        cat = row["Category"]
        spent = row[amount_col]

        if spent <= 0:
            savings[cat] = 0
            continue

        if spent < 3000:
            base_ratio = 0.07
        elif spent < 15000:
            base_ratio = 0.15
        else:
            base_ratio = 0.25

        weight = spent / total_expense
        savings[cat] = round(spent * base_ratio * (0.5 + weight), 0)

    savings_df = pd.DataFrame({
        "Category": savings.keys(),
        "Potential Savings": savings.values()
    })

    # --------------------------------------------------
    # SAVINGS PIE CHART
    # --------------------------------------------------
    st.subheader("💡 Predicted Potential Savings Distribution")

    pie = alt.Chart(savings_df).mark_arc(innerRadius=60).encode(
        theta="Potential Savings",
        color="Category",
        tooltip=["Category", "Potential Savings"]
    )

    st.altair_chart(pie, use_container_width=True)

# --------------------------------------------------
# AI CHATBOT 
# --------------------------------------------------
st.subheader("🤖 Financial Advisor Chat")

if "last_q" not in st.session_state:
    st.session_state.last_q = ""
    st.session_state.last_a = ""

user_input = st.text_input("Ask about savings, spending or investments")

if user_input:
    client = Groq(api_key=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY")))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a smart Indian financial advisor."},
            {"role": "user", "content": user_input}
        ]
    )

    st.session_state.last_a = response.choices[0].message.content.strip()

if st.session_state.last_a:
    box = st.empty()
    text = ""
    for ch in st.session_state.last_a:
        text += ch
        box.markdown(text)
        time.sleep(0.01)
