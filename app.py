import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation App")
st.write("Enter customer details to predict the customer segment.")

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "kmeans_pipeline.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()

    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

# -----------------------------
# Input Form
# -----------------------------
with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        income = st.number_input(
            "Income",
            min_value=0,
            value=50000,
            step=1000
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        city = st.selectbox(
            "City",
            [
                "Indore",
                "Bhopal",
                "Delhi",
                "Mumbai",
                "Pune"
            ]
        )

    with col2:
        purchase_frequency = st.number_input(
            "Purchase Frequency",
            min_value=1,
            max_value=100,
            value=5
        )

        last_purchase_days = st.number_input(
            "Last Purchase Days",
            min_value=0,
            value=30
        )

        spending_score = st.number_input(
            "Spending Score",
            min_value=1,
            max_value=100,
            value=50
        )

        product_category = st.selectbox(
            "Product Category",
            [
                "Electronics",
                "Fashion",
                "Grocery",
                "Sports",
                "Beauty"
            ]
        )

        membership_status = st.selectbox(
            "Membership Status",
            [
                "Basic",
                "Silver",
                "Gold"
            ]
        )

    submitted = st.form_submit_button(
        "Predict Customer Cluster"
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:

    try:

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

        st.subheader("Customer Information")
        st.dataframe(input_df, use_container_width=True)

        cluster = model.predict(input_df)[0]

        cluster_names = {
            0: "Low Value Inactive Customers",
            1: "Occasional Regular Customers",
            2: "High Income Underutilized VIPs",
            3: "Active High Value Customers"
        }

        cluster_name = cluster_names.get(
            cluster,
            f"Cluster {cluster}"
        )

        st.success(
            f"✅ Customer belongs to: {cluster_name}"
        )

        st.info(
            f"Cluster Number: {cluster}"
        )

    except Exception as e:
        st.error(
            f"❌ Prediction failed: {e}"
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Customer Segmentation using KMeans Clustering")