import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

# -----------------------------
# Helper Functions
# -----------------------------
DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"]

def try_parse_date(s: str):
    if pd.isna(s):
        return None
    s = str(s).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return None

def categorize(desc: str) -> str:
    desc = str(desc).lower()
    if "food" in desc or "restaurant" in desc or "cafe" in desc:
        return "Food"
    elif "grocery" in desc or "supermarket" in desc:
        return "Groceries"
    elif "hospital" in desc or "pharmacy" in desc:
        return "Health"
    elif "movie" in desc or "netflix" in desc or "spotify" in desc:
        return "Entertainment"
    elif "amazon" in desc or "flipkart" in desc or "myntra" in desc:
        return "Shopping"
    elif "electricity" in desc or "water" in desc or "bill" in desc:
        return "Bills"
    elif "balance" in desc:
        return "Balance Forward"
    else:
        return "Other"

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="💰 Smart Finance Assistant", layout="wide")

st.title("💰 Smart Finance Assistant")
st.write("Upload your bank transactions CSV and get deep insights!")

uploaded_file = st.file_uploader("📤 Upload Transactions CSV", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(df_raw.head())

    # Detect Columns
    date_col = None
    amount_col = None
    desc_col = None
    type_col = None

    for col in df_raw.columns:
        if "date" in col.lower():
            date_col = col
        if "amount" in col.lower():
            amount_col = col
        if "desc" in col.lower() or "narration" in col.lower():
            desc_col = col
        if "type" in col.lower() or "debit" in col.lower() or "credit" in col.lower():
            type_col = col

    if not all([date_col, amount_col, desc_col]):
        st.error("❌ Could not auto-detect necessary columns. Please ensure CSV has Date, Amount, and Description columns.")
    else:
        # Parse dates safely
        df_raw[date_col] = df_raw[date_col].apply(try_parse_date)
        df_raw = df_raw.dropna(subset=[date_col])
        df_raw[amount_col] = df_raw[amount_col].astype(str).str.replace(",", "", regex=False)
        df_raw[amount_col] = pd.to_numeric(df_raw[amount_col], errors="coerce").fillna(0)

        # Categorize transactions
        df_raw["Category"] = df_raw[desc_col].apply(categorize)

        # Extract opening balance if exists
        opening_balance = 0
        if "balance" in df_raw[type_col].str.lower().values:
            ob_row = df_raw[df_raw[type_col].str.lower() == "balance"]
            if not ob_row.empty:
                opening_balance = ob_row.iloc[0][amount_col]
                df_raw = df_raw[df_raw[type_col].str.lower() != "balance"]

        # Normalize income vs expense
        if type_col:
            df_raw["Transaction"] = df_raw[type_col].str.lower().map(
                lambda x: "Income" if "credit" in x else ("Expense" if "debit" in x else "Other")
            )
        else:
            # Fallback if type not given: assume +ve = income, -ve = expense
            df_raw["Transaction"] = df_raw[amount_col].apply(lambda x: "Income" if x > 0 else "Expense")

        # -----------------------------
        # Filters
        # -----------------------------
        with st.expander("🔎 Filters", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                min_date = df_raw[date_col].min()
                max_date = df_raw[date_col].max()
                date_range = st.date_input(
                    "📅 Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date = end_date = date_range
                df_raw = df_raw[
                    (pd.to_datetime(df_raw[date_col], errors="coerce") >= pd.to_datetime(start_date)) &
                    (pd.to_datetime(df_raw[date_col], errors="coerce") <= pd.to_datetime(end_date))
                ]

            with col2:
                amount_filter = st.selectbox(
                    "💵 Filter by Amount",
                    ["All", "< 500", "< 1000", "< 5000", "< 10000"]
                )
                if amount_filter != "All":
                    threshold = int("".join([ch for ch in amount_filter if ch.isdigit()]))
                    df_raw = df_raw[df_raw[amount_col].abs() < threshold]

            with col3:
                if type_col:
                    type_filter = st.selectbox(
                        "🔄 Filter by Type",
                        ["All"] + list(df_raw["Transaction"].unique())
                    )
                    if type_filter != "All":
                        df_raw = df_raw[df_raw["Transaction"] == type_filter]

        # Category filter
        categories = ["Food", "Groceries", "Health", "Entertainment", "Shopping", "Bills", "Other"]
        category_filter = st.multiselect("📂 Filter by Category", options=categories, default=categories)
        df_raw = df_raw[df_raw["Category"].isin(category_filter)]

        # -----------------------------
        # Insights
        # -----------------------------
        st.subheader("📈 Insights")

        total_income = df_raw[df_raw["Transaction"] == "Income"][amount_col].sum()
        total_expense = df_raw[df_raw["Transaction"] == "Expense"][amount_col].sum()
        net_savings = (opening_balance + total_income) - total_expense

        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Total Income", f"₹{total_income:,.2f}")
        c2.metric("💸 Total Expenses", f"₹{total_expense:,.2f}")
        c3.metric("🏦 Net Balance", f"₹{net_savings:,.2f}")

        # -----------------------------
        # Charts
        # -----------------------------
        st.subheader("📊 Visualizations")

        # Expenses by Category
        expense_by_cat = df_raw[df_raw["Transaction"] == "Expense"].groupby("Category")[amount_col].sum().reset_index()
        if not expense_by_cat.empty:
            chart1 = alt.Chart(expense_by_cat).mark_bar().encode(
                x=alt.X("Category", sort="-y"),
                y=alt.Y(amount_col, title="Total Spent"),
                color="Category"
            )
            st.altair_chart(chart1, use_container_width=True)

        # Trend Over Time
        trend = df_raw.groupby(date_col)[amount_col].sum().reset_index()
        if not trend.empty:
            chart2 = alt.Chart(trend).mark_line(point=True).encode(
                x=date_col,
                y=amount_col,
                tooltip=[date_col, amount_col]
            )
            st.altair_chart(chart2, use_container_width=True)

        # -----------------------------
        # Show Final Table
        # -----------------------------
        st.subheader("📂 Transactions")
        st.dataframe(df_raw)

        # -----------------------------
        # Smart Summary & Advice
        # -----------------------------
        st.subheader("🧾 Personalized Summary & Advice")

        if not expense_by_cat.empty:
            max_cat = expense_by_cat.loc[expense_by_cat[amount_col].idxmax()]["Category"]
            advice = []

            if total_expense > total_income * 0.7:
                advice.append("⚠️ Your expenses are very high compared to income. Try to cut down on discretionary spending.")
            else:
                advice.append("✅ Your expense ratio looks healthy. Keep maintaining this balance!")

            advice.append(f"💡 You spent the most on **{max_cat}**. Consider setting a budget or tracking this category closely.")

            if net_savings > 0:
                advice.append("📈 You have positive savings. Consider investing part of it in SIPs, index funds, or emergency savings.")
            else:
                advice.append("🚨 You're overspending and running into negative savings. Reduce costs in non-essential categories.")

            for tip in advice:
                st.markdown(tip)

else:
    st.info("📤 Please upload a CSV file to continue.")
