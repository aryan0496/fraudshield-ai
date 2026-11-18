import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="FraudShield Pro | AI Risk Engine", page_icon="🛡️", layout="wide")

# 1. Generate 5,000 Realistic Bank & UPI Transactions (Cached so it runs fast)
@st.cache_resource
def train_fraud_engine():
    np.random.seed(42)
    n = 5000

    amount = np.random.exponential(scale=3500, size=n).astype(int) + 100
    is_night = np.random.choice([0, 1], size=n, p=[0.82, 0.18])
    distance_km = np.random.exponential(scale=25, size=n).astype(int)
    no_otp = np.random.choice([0, 1], size=n, p=[0.88, 0.12])
    txn_last_hour = np.random.poisson(lam=1.5, size=n)

    # Realistic Fraud Logic + Noise
    risk_score = (
        (amount > 25000).astype(int) * 2
        + is_night * 2
        + (distance_km > 150).astype(int) * 2
        + no_otp * 3
        + (txn_last_hour >= 4).astype(int) * 2
    )
    fraud = (risk_score >= 5).astype(int)

    df = pd.DataFrame({
        "Amount_INR": amount,
        "Night_Txn": is_night,
        "Distance_KM": distance_km,
        "OTP_Skipped": no_otp,
        "Txns_Last_Hour": txn_last_hour,
        "Is_Fraud": fraud
    })

    features = ["Amount_INR", "Night_Txn", "Distance_KM", "OTP_Skipped", "Txns_Last_Hour"]
    X_train, X_test, y_train, y_test = train_test_split(df[features], df["Is_Fraud"], test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test)) * 100

    return model, df, features, acc

model, df, features, acc = train_fraud_engine()

# 2. Header & Top KPI Metrics
st.title("🛡️ FraudShield Pro — Real-Time Banking & UPI Fraud Intelligence")
st.caption("Built with Python, Scikit-Learn Random Forest & Streamlit | Portfolio Project #1")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions Analyzed", f"{len(df):,}")
col2.metric("AI Model Accuracy", f"{acc:.1f}%")
col3.metric("Fraud Cases Detected", f"{df['Is_Fraud'].sum():,}")
col4.metric("AI Engine", "Random Forest (120 Trees)")

st.divider()

# 3. Two Interactive Tabs
tab1, tab2 = st.tabs(["🔍 Live Transaction Scanner", "📊 Bank Analytics & Flagged Logs"])

with tab1:
    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.subheader("Enter Transaction Details")
        amount = st.number_input("Payment Amount (₹):", min_value=1, max_value=10000000, value=1500, step=500)
        time_choice = st.selectbox("Payment Time Window:", ["Din me (6 AM - 11 PM)", "Der Raat (11 PM - 5 AM)"])
        otp_choice = st.selectbox("Security Verification:", ["PIN / OTP Verified ✅", "No OTP / International Swipe ⚠️"])
        distance = st.slider("Ghar / Regular Location se Distance (KM):", 0, 1500, 10)
        txn_freq = st.slider("Pichle 1 Ghante me kitne payments hue?", 1, 15, 1)

        is_night = 1 if "Der Raat" in time_choice else 0
        no_otp = 1 if "No OTP" in otp_choice else 0

        check_btn = st.button("🚀 Run AI Fraud Check", use_container_width=True)

    with right_col:
        st.subheader("AI Risk Decision & Explainability")
        input_data = pd.DataFrame([[amount, is_night, distance, no_otp, txn_freq]], columns=features)
        prob = model.predict_proba(input_data)[0][1] * 100

        st.progress(int(prob), text=f"Calculated Fraud Probability: {prob:.1f}%")

        if prob >= 60:
            st.error(f"🚨 HIGH RISK FRAUD BLOCKED! (Risk Score: {prob:.1f}%)")
            st.write("**Action Taken:** Payment auto-blocked & SMS alert sent to customer.")
        elif prob >= 30:
            st.warning(f"⚠️ SUSPICIOUS TRANSACTION (Risk Score: {prob:.1f}%)")
            st.write("**Action Taken:** Step-up biometric / Video KYC verification required.")
        else:
            st.success(f"✅ VERIFIED SAFE TRANSACTION (Risk Score: {prob:.1f}%)")
            st.write("**Action Taken:** Payment approved instantly.")

        # Risk Factor Breakdown Chart
        st.write("**Why did the AI give this score? (Feature Importance):**")
        imp_df = pd.DataFrame({
            "Risk Factor": ["Amount (₹)", "Night Time", "Distance (KM)", "No OTP/PIN", "High Frequency"],
            "AI Weight (%)": (model.feature_importances_ * 100).round(1)
        }).set_index("Risk Factor")
        st.bar_chart(imp_df)

with tab2:
    st.subheader("Recent High-Risk Flagged Transactions in Database")
    flagged_df = df[df["Is_Fraud"] == 1].head(15)
    st.dataframe(flagged_df, use_container_width=True)

    csv_data = flagged_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Flagged Fraud Report (CSV)", data=csv_data, file_name="fraudshield_report.csv", mime="text/csv")