"""
app.py  -  Customer Churn Prediction Dashboard
Run with:  streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
    .section-title { font-size: 1.15rem; font-weight: 600; margin-bottom: .4rem; }
    .risk-high   { background:#fee2e2; border-left:4px solid #ef4444;
                   padding:1rem 1.2rem; border-radius:6px; }
    .risk-medium { background:#fef9c3; border-left:4px solid #eab308;
                   padding:1rem 1.2rem; border-radius:6px; }
    .risk-low    { background:#dcfce7; border-left:4px solid #22c55e;
                   padding:1rem 1.2rem; border-radius:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load data & model (cached) ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/customer_churn.csv")
    return df

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        artefact = pickle.load(f)
    return artefact

df      = load_data()
artefact = load_model()
model    = artefact["model"]
model_name = artefact["model_name"]

# ── Encode helpers ────────────────────────────────────────────────────────────
def encode_row(gender, senior, tenure, monthly, total, contract, tech):
    gender_enc    = 1 if gender == "Male" else 0
    le_contract   = LabelEncoder()
    le_contract.fit(["Month-to-month", "One year", "Two year"])
    contract_enc  = int(le_contract.transform([contract])[0])
    tech_enc      = 1 if tech == "Yes" else 0
    return np.array([[gender_enc, senior, tenure, monthly, total, contract_enc, tech_enc]])

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["📊  Dataset Overview & Analytics", "🔮  Churn Prediction"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Dataset Overview & Analytics
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.title("📡 Customer Churn — Dataset Overview")
    st.caption(f"Synthetic telecom dataset · {len(df):,} customers · Model in use: **{model_name}**")

    # ── KPI row ──────────────────────────────────────────────────────────────
    total_customers  = len(df)
    churn_rate       = (df["Churn"] == "Yes").mean() * 100
    avg_monthly      = df["MonthlyCharges"].mean()
    avg_tenure       = df["Tenure_Months"].mean()
    churned_count    = (df["Churn"] == "Yes").sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers",   f"{total_customers:,}")
    k2.metric("Churn Rate",        f"{churn_rate:.1f}%")
    k3.metric("Avg Monthly Charge",f"${avg_monthly:.2f}")
    k4.metric("Avg Tenure",        f"{avg_tenure:.1f} mo")

    st.divider()

    # ── Row 1: Churn distribution  +  Contract-type breakdown ────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Churn Distribution</p>', unsafe_allow_html=True)
        churn_counts = df["Churn"].value_counts()
        fig, ax = plt.subplots(figsize=(4.5, 4))
        colors = ["#22c55e", "#ef4444"]
        wedges, texts, autotexts = ax.pie(
            churn_counts,
            labels=churn_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops=dict(edgecolor="white", linewidth=2),
        )
        for t in autotexts:
            t.set_fontsize(11)
            t.set_fontweight("bold")
        ax.set_title("Churn vs Retained", fontsize=12, fontweight="bold", pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown('<p class="section-title">Churn Rate by Contract Type</p>', unsafe_allow_html=True)
        contract_churn = (
            df.groupby("ContractType")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .rename("ChurnRate")
            .reset_index()
        )
        contract_churn = contract_churn.sort_values("ChurnRate", ascending=True)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        bars = ax.barh(
            contract_churn["ContractType"],
            contract_churn["ChurnRate"],
            color=["#3b82f6", "#f59e0b", "#ef4444"],
            edgecolor="white",
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
        ax.set_xlabel("Churn Rate (%)")
        ax.set_title("Contract Type vs Churn Rate", fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_xlim(0, contract_churn["ChurnRate"].max() * 1.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Row 2: Monthly charges dist  +  Tenure vs Churn ──────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<p class="section-title">Monthly Charges Distribution by Churn</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        for label, color in [("No", "#22c55e"), ("Yes", "#ef4444")]:
            subset = df[df["Churn"] == label]["MonthlyCharges"]
            ax.hist(subset, bins=25, alpha=0.65, label=label, color=color, edgecolor="white")
        ax.set_xlabel("Monthly Charges ($)")
        ax.set_ylabel("Count")
        ax.set_title("Monthly Charges by Churn Status", fontsize=12, fontweight="bold")
        ax.legend(title="Churn")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col4:
        st.markdown('<p class="section-title">Avg Tenure by Contract & Churn</p>', unsafe_allow_html=True)
        tenure_data = df.groupby(["ContractType", "Churn"])["Tenure_Months"].mean().reset_index()
        pivot = tenure_data.pivot(index="ContractType", columns="Churn", values="Tenure_Months")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        pivot.plot(kind="bar", ax=ax, color=["#22c55e", "#ef4444"], edgecolor="white", width=0.6)
        ax.set_xlabel("")
        ax.set_ylabel("Avg Tenure (Months)")
        ax.set_title("Avg Tenure by Contract & Churn", fontsize=12, fontweight="bold")
        ax.legend(title="Churn")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Row 3: Tech support vs churn + senior citizen ─────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<p class="section-title">Tech Support Impact on Churn</p>', unsafe_allow_html=True)
        tech_churn = (
            df.groupby("TechSupport")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .rename("ChurnRate")
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(4.5, 4))
        bars = ax.bar(
            tech_churn["TechSupport"],
            tech_churn["ChurnRate"],
            color=["#ef4444", "#22c55e"],
            edgecolor="white",
            width=0.45,
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11, fontweight="bold")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Tech Support vs Churn Rate", fontsize=12, fontweight="bold")
        ax.set_ylim(0, tech_churn["ChurnRate"].max() * 1.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col6:
        st.markdown('<p class="section-title">Senior Citizen vs Churn</p>', unsafe_allow_html=True)
        senior_churn = (
            df.groupby("SeniorCitizen")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .rename("ChurnRate")
            .reset_index()
        )
        senior_churn["Label"] = senior_churn["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
        fig, ax = plt.subplots(figsize=(4.5, 4))
        bars = ax.bar(
            senior_churn["Label"],
            senior_churn["ChurnRate"],
            color=["#3b82f6", "#f59e0b"],
            edgecolor="white",
            width=0.45,
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11, fontweight="bold")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Senior Citizen vs Churn Rate", fontsize=12, fontweight="bold")
        ax.set_ylim(0, senior_churn["ChurnRate"].max() * 1.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Raw data table ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("View Raw Dataset"):
        st.dataframe(df, use_container_width=True, height=340)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Interactive Churn Prediction
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.title("🔮 Churn Risk Predictor")
    st.caption("Fill in the customer profile below and click **Predict** to get a churn risk score.")

    with st.form("prediction_form"):
        st.markdown("### Customer Profile")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            gender        = st.selectbox("Gender",           ["Male", "Female"])
            senior        = st.selectbox("Senior Citizen",   [0, 1],
                                         format_func=lambda x: "Yes" if x == 1 else "No")
            tech_support  = st.selectbox("Tech Support",     ["Yes", "No"])
        with fc2:
            contract      = st.selectbox("Contract Type",    ["Month-to-month", "One year", "Two year"])
            tenure        = st.slider("Tenure (Months)",      1, 72, 12)
        with fc3:
            monthly       = st.number_input("Monthly Charges ($)",  min_value=10.0, max_value=150.0,
                                             value=65.0, step=0.5, format="%.2f")
            total         = st.number_input("Total Charges ($)",    min_value=0.0,  max_value=10000.0,
                                             value=float(round(65.0 * 12, 2)), step=1.0, format="%.2f")

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True, type="primary")

    if submitted:
        X_input = encode_row(gender, senior, tenure, monthly, total, contract, tech_support)
        prob        = model.predict_proba(X_input)[0][1]
        prob_pct    = prob * 100

        # Risk categorisation
        if prob_pct >= 60:
            risk_label = "HIGH RISK"
            risk_class = "risk-high"
            risk_emoji = "🔴"
        elif prob_pct >= 35:
            risk_label = "MEDIUM RISK"
            risk_class = "risk-medium"
            risk_emoji = "🟡"
        else:
            risk_label = "LOW RISK"
            risk_class = "risk-low"
            risk_emoji = "🟢"

        st.divider()
        st.markdown("### Prediction Result")

        r1, r2 = st.columns([2, 1])
        with r1:
            st.markdown(
                f"""
                <div class="{risk_class}">
                  <h2 style="margin:0">{risk_emoji} {risk_label}</h2>
                  <p style="margin:.4rem 0 0 0;font-size:1.1rem">
                    Churn Probability: <strong>{prob_pct:.1f}%</strong>
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r2:
            # Gauge-style donut chart
            fig, ax = plt.subplots(figsize=(3, 3))
            gauge_colors = ["#ef4444" if prob_pct >= 60 else "#eab308" if prob_pct >= 35 else "#22c55e", "#e5e7eb"]
            ax.pie(
                [prob_pct, 100 - prob_pct],
                colors=gauge_colors,
                startangle=90,
                wedgeprops=dict(width=0.45, edgecolor="white"),
            )
            ax.text(0, 0, f"{prob_pct:.1f}%", ha="center", va="center",
                    fontsize=16, fontweight="bold")
            ax.set_title("Churn Probability", fontsize=10, fontweight="bold")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ── Factor summary ────────────────────────────────────────────────────
        st.markdown("#### Key Risk Factors for this Customer")
        factors = {
            "Contract Type":   ("High" if contract == "Month-to-month" else "Low",
                                contract == "Month-to-month"),
            "Tenure":          ("Short (<= 12 mo)" if tenure <= 12 else "Long (> 12 mo)",
                                tenure <= 12),
            "Tech Support":    ("No (higher risk)" if tech_support == "No" else "Yes (lower risk)",
                                tech_support == "No"),
            "Senior Citizen":  ("Yes (higher risk)" if senior == 1 else "No",
                                senior == 1),
            "Monthly Charges": (f"${monthly:.2f} ({'high' if monthly > 70 else 'moderate/low'})",
                                monthly > 70),
        }
        for factor, (detail, is_risk) in factors.items():
            icon = "⚠️" if is_risk else "✅"
            st.markdown(f"- {icon} **{factor}**: {detail}")

        st.info(
            f"Model used: **{model_name}** · This prediction is based on a synthetic dataset "
            "and is intended for demonstration purposes only.",
            icon="ℹ️",
        )
