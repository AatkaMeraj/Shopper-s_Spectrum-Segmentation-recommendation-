import streamlit as st
import pandas as pd
import pickle
import requests
import os
import gdown
import numpy as np
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation & Recommendation", layout="wide")

# --------- File IDs and Filenames --------- #
FILE_IDS = {
    "kmeans_model.pkl": "16xyOUF8GPwl8R2NU-0JSKHFxiOKJ_BbZ",
    "scaler.pkl": "1edsz2jUstqY-vGW5hAgO_n5uQOzXGExY",
    "user_item_matrix.pkl": "14LL-3Pw1AHLJgvB4YNaZaUOCuW3R83Fz",
    "user_sim_df.pkl": "1azW9ip00mg01na-VLyf7de0dYvehtow7",
    "item_sim_df.pkl": "1Ksx1ve8fC9xVRfhasF_4Dg9EySohx0BL",
}

# --------- Utility to Download and Load Pickle Files --------- #
@st.cache_resource
def download_and_load_pickle(file_id, filename):
    if not os.path.exists(filename):
        try:
            gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)
        except Exception as e:
            raise Exception(f"Download failed: {e}")

    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception(f"Error loading pickle file '{filename}': {e}")

# --------- Try Loading Files --------- #
try:
    kmeans = download_and_load_pickle(FILE_IDS["kmeans_model.pkl"], "kmeans_model.pkl")
    scaler = download_and_load_pickle(FILE_IDS["scaler.pkl"], "scaler.pkl")
    user_item_matrix = download_and_load_pickle(FILE_IDS["user_item_matrix.pkl"], "user_item_matrix.pkl")
    user_sim_df = download_and_load_pickle(FILE_IDS["user_sim_df.pkl"], "user_sim_df.pkl")
    item_sim_df = download_and_load_pickle(FILE_IDS["item_sim_df.pkl"], "item_sim_df.pkl")
except Exception as e:
    st.error("❌ Failed to load files:")
    st.exception(e)
    st.stop()
# ----------- Segment Prediction ----------- #
def predict_segment(r, f, m, customer_id):
    dummy_customer_id = 99999
    data = scaler.transform([[r, f, m, dummy_customer_id]])
    segment = kmeans.predict(data)[0]
    labels = {
        0: "Loyal Customer",
        1: "Occasional Customer",
        2: "Lost Customer",
        3: "Big Spender",
    }
    return segment, labels.get(segment, "Unknown")

# ----------- Recommendation Functions ----------- #
def predict_user_user(customer_id, product_desc):
    if product_desc not in user_item_matrix.columns:
        return None
    sim_scores = user_sim_df[customer_id]
    item_ratings = user_item_matrix[product_desc]
    mask = (item_ratings > 0) & (user_item_matrix.index != customer_id)
    if not mask.any():
        return None
    numerator = (sim_scores[mask] * item_ratings[mask]).sum()
    denominator = sim_scores[mask].sum()
    return numerator / denominator if denominator != 0 else None

def predict_item_item(customer_id, product_desc):
    if product_desc not in user_item_matrix.columns:
        return None
    user_ratings = user_item_matrix.loc[customer_id]
    sim_scores = item_sim_df[product_desc]
    mask = (user_ratings > 0) & (user_item_matrix.columns != product_desc)
    if not mask.any():
        return None
    numerator = (sim_scores[mask] * user_ratings[mask]).sum()
    denominator = sim_scores[mask].sum()
    return numerator / denominator if denominator != 0 else None

# ----------- UI: Page Navigation ----------- #
page = st.sidebar.radio("Select Page", ["Home", "Customer Segmentation", "Product Recommendation"])

# ----------- Home Page ----------- #
if page == "Home":
    st.title("🛒 Shopper's Spectrum Dashboard")
    st.markdown("Welcome! Use the sidebar to explore segmentation and product recommendation features.")

# ----------- Customer Segmentation Page ----------- #
elif page == "Customer Segmentation":
    st.title("Customer Segmentation")
    st.markdown("Enter customer RFM values to predict segment.")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=325)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=1)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=765322.0)

    if st.button("Predict Segment"):
        seg_id, label = predict_segment(recency, frequency, monetary, customer_id=99999)
        st.success(f"Cluster No: {seg_id}")
        st.write(f"This customer belongs to: **{label}**")

# ----------- Product Recommendation Page ----------- #
elif page == "Product Recommendation":
    st.title("Product Recommendation")
    st.markdown("Select a product to get similar product recommendations based on Collaborative Filtering.")

    selected_product = st.selectbox("Select a Product", user_item_matrix.columns)
    top_n = st.slider("Number of Recommendations", 3, 15, 5)

    if st.button("Generate Recommendations"):
        if selected_product not in item_sim_df.columns:
            st.warning("Selected product not found in similarity data.")
        else:
            sim_scores = item_sim_df[selected_product].drop(selected_product)  # Exclude self
            top_similar = sim_scores.sort_values(ascending=False).head(top_n)

            st.subheader(f"Top {top_n} products similar to '{selected_product}':")
            for i, (prod, score) in enumerate(top_similar.items(), 1):
                st.write(f"{i}. {prod}")






