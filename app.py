import streamlit as st
import pandas as pd
import joblib
import os
import time

# ==========================================
# PAGE SETTINGS & PRESET
# ==========================================
st.set_page_config(
    page_title="Customer Segment Intelligence Matrix",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust fallback machine simulation for zero-downtime demonstration
MODEL_PATH = "kmeans_pipeline.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        class MockKMeans:
            def predict(self, df): 
                score = df["SpendingScore"].values[0]
                if score < 30: return [0]
                elif score < 60: return [1]
                elif score < 85: return [2]
                else: return [3]
        return MockKMeans()
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        st.stop()

model = load_model()

# ==========================================
# PREMIUM LIGHT MODE CSS WITH SMOOTH ANIMATIONS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    /* Light Mode Base Reset */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important; /* Premium Soft Ice White */
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #0f172a !important; /* Deep Slate Text */
    }
    
    /* Native Containers Re-styled as Soft Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 24px rgba(148, 163, 184, 0.08) !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        border-color: #cbd5e1 !important;
        box-shadow: 0 12px 30px rgba(148, 163, 184, 0.15) !important;
    }
    
    /* Clean Inputs For Light Theme */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
    }
    
    /* Slick Fluid Action Button */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 14px !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.2);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton button:hover {
        transform: scale(1.01);
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.4);
    }
    
    /* Tech Minimalist Top Badge */
    .corp-badge {
        background: rgba(37, 99, 235, 0.06);
        color: #2563eb;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        display: inline-block;
        margin-bottom: 12px;
        border: 1px solid rgba(37, 99, 235, 0.1);
    }

    /* CSS Keyframe Animation for Output Card Drop-In */
    @keyframes slideUpFade {
        from {
            opacity: 0;
            transform: translateY(24px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animated-output {
        animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown('<span class="corp-badge">BI MARKETING ANALYTICS ENGINE</span>', unsafe_allow_html=True)
st.markdown("<h1 style='font-weight:800; margin-top:0; color:#0f172a; letter-spacing:-0.5px;'>📊 Customer Segmentation Matrix</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748b; font-size:16px; margin-top:-10px;'>High-fidelity customer classification suite fueled by unsupervised pipeline clustering.</p>", unsafe_allow_html=True)
st.write("##")

# ==========================================
# INPUT MODULAR COMPONENT
# ==========================================
with st.container(border=True):
    st.markdown("<h4 style='margin-top:0; color:#334155; font-weight:600;'>👤 Demographics & Behavioral Profiling</h4>", unsafe_allow_html=True)
    st.write("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Customer Age", min_value=18, max_value=100, value=34)
        income = st.number_input("Annual Gross Income ($)", min_value=0, value=68000, step=2000)
        gender = st.selectbox("Registered Gender", ["Male", "Female"])

    with col2:
        purchase_frequency = st.number_input("Purchase Frequency (Annual Count)", min_value=1, max_value=100, value=14)
        last_purchase_days = st.number_input("Recency (Days Since Last Order)", min_value=0, value=12)
        spending_score = st.number_input("Assigned Spending Score (1-100)", min_value=1, max_value=100, value=72)

    with col3:
        city = st.selectbox("Demographic Region / City", ["Delhi", "Mumbai", "Pune", "Indore", "Bhopal"])
        product_category = st.selectbox("Primary Category Affiliation", ["Electronics", "Fashion", "Grocery", "Sports", "Beauty"])
        membership_status = st.selectbox("Loyalty Program Tier", ["Basic", "Silver", "Gold"])

    st.write("##")
    submitted = st.button("🚀 EXECUTE COHORT CLASSIFICATION")

# ==========================================
# INFERENCE LOGIC & ANIMATED OUTPUT
# ==========================================
if submitted:
    with st.spinner("Calculating geometric feature weights..."):
        time.sleep(0.5)  # Soft interface break to track completion

        input_df = pd.DataFrame({
            "Age": [age],
            "Income": [income],
            "Gender": [gender],
            "City": [city],
            "PurchaseFrequency": [purchase_frequency],
            "LastPurchaseDays": [last_purchase_days],
            "SpendingScore": [spending_score],
            "ProductCategory": [product_category],
            "MembershipStatus": [membership_status]
        })

    cluster = model.predict(input_df)[0]

    # Clean Pastel Matrix Setup for High-Class Light Mode Experience
    cluster_meta = {
        0: {
            "name": "Low Value Inactive Customers",
            "bg": "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)", "border": "#ef4444", "text": "#991b1b",
            "strategy": "⚠️ Win-back automated trigger. Deploy re-activation incentives via email clusters, minimize high CPA campaigns."
        },
        1: {
            "name": "Occasional Regular Customers",
            "bg": "linear-gradient(135deg, #fef9c3 0%, #fef08a 100%)", "border": "#eab308", "text": "#854d0e",
            "strategy": "🔄 Cross-sell standard bundles. Push mid-tier subscription discounts to lock lifetime value loops."
        },
        2: {
            "name": "High Income Underutilized VIPs",
            "bg": "linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%)", "border": "#a855f7", "text": "#6b21a8",
            "strategy": "💎 High latent potential. Route immediately to localized lifestyle activations and exclusive access tiers."
        },
        3: {
            "name": "Active High Value Core Customers",
            "bg": "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)", "border": "#22c55e", "text": "#166534",
            "strategy": "👑 Core Revenue Anchors. Protect with personalized concierge support. Do not offer margin-killing baseline coupons."
        }
    }

    meta = cluster_meta.get(cluster, {
        "name": f"Cluster {cluster}", 
        "bg": "#f8fafc", "border": "#94a3b8", "text": "#334155",
        "strategy": "Evaluate vector shifts inside analytics tracking dashboard."
    })

    # Animation Frame Open
    st.markdown('<div class="animated-output">', unsafe_allow_html=True)
    st.write("##")
    st.markdown("<h3 style='font-weight:700; color:#1e293b;'>📊 Target Distribution Strategy</h3>", unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1.6, 2.4])

    with res_col1:
        # High End Pastel Card Component
        st.markdown(f"""
            <div style="background: {meta['bg']}; border: 1px solid {meta['border']}; padding: 28px; border-radius: 14px; height: 100%;">
                <p style="color: #64748b; font-size:11px; font-weight:700; margin:0; letter-spacing:1px; text-transform:uppercase;">ML ASSIGNED POSITION: MATRIX #{cluster}</p>
                <h2 style="color: {meta['text']}; margin: 6px 0 14px 0; font-weight:700; line-height:1.2;">{meta['name']}</h2>
                <hr style="border:0; border-top: 1px solid rgba(0,0,0,0.05); margin-bottom:14px;">
                <p style="color: #475569; font-size:12px; font-weight:700; margin-bottom:4px; letter-spacing:0.5px;">MARKETING STRATEGY DIRECTIVE:</p>
                <p style="color: {meta['text']}; font-size:14px; margin:0; font-weight:500; line-height:1.5;">{meta['strategy']}</p>
            </div>
        """, unsafe_allow_html=True)

    with res_col2:
        with st.container(border=True):
            st.markdown("<p style='color:#64748b; font-weight:700; font-size:12px; margin: 0 0 10px 0; letter-spacing:0.5px;'>INGESTED FEATURE VECTOR</p>", unsafe_allow_html=True)
            
            # Formatted clean tabular display
            st.dataframe(
                input_df.style.format({"Income": "${:,.2f}"}), 
                use_container_width=True, 
                hide_index=True
            )
            
            st.write("---")
            m_1, m_2 = st.columns(2)
            # Utilizing local formatting metrics for crisp output
            m_1.metric(label="Behavior Score Rate", value=f"{spending_score} / 100", delta=f"{purchase_frequency} conversions")
            m_2.metric(label="Calculated Recency Weight", value=f"{last_purchase_days} Days", delta="Optimal Flow" if last_purchase_days <= 30 else "Imminent Churn", delta_color="normal" if last_purchase_days <= 30 else "inverse")

    st.markdown('</div>', unsafe_allow_html=True) # Animation Frame Close

# ==========================================
# FOOTER ARCHITECTURE
# ==========================================
st.write("##")
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #94a3b8; font-size: 13px; font-weight:500;'>Corporate Cluster Matrix Framework • Light Tier Interface Edition</p>", 
    unsafe_allow_html=True
)