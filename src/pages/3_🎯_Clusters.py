import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data, perform_clustering
from utils.visualizations import plot_cluster_3d, plot_radar_chart, plot_histogram

st.set_page_config(page_title="🎯 Lung Cancer Clusters", page_icon="🎯", layout="wide")

# =========================
# Load Data
# =========================
df = load_data()

# Filter only patients with cancer
df_cancer = df[df['lung_cancer'] == 1].copy()
if df_cancer.empty:
    st.warning("No cancer patients found in the dataset.")
    st.stop()

# =========================
# Page Title
# =========================
st.title("🎯 Lung Cancer Clusters Analysis")

# =========================
# Clustering Parameters
# =========================
n_clusters = st.slider("Number of clusters", min_value=2, max_value=6, value=3)

# Define features for clustering
features = ['age_scaled', 'gender', 'smoking', 'yellow_fingers', 'anxiety', 'peer_pressure',
            'chronic_disease', 'fatigue', 'allergy', 'wheezing',
            'alcohol_consuming', 'coughing', 'shortness_of_breath',
            'swallowing_difficulty', 'chest_pain']

# =========================
# Perform Clustering
# =========================
clusters, X_pca, cluster_profiles = perform_clustering(df_cancer, features=features, n_clusters=n_clusters)
df_cancer['cluster'] = clusters.astype(str)  # For coloring and display

# =========================
# 3D PCA Cluster Plot
# =========================
st.subheader("🔹 3D Cluster Visualization (PCA)")
fig_3d = plot_cluster_3d(X_pca, clusters)
st.plotly_chart(fig_3d, use_container_width=True)

# =========================
# Cluster Radar Profiles
# =========================
st.subheader("📊 Cluster Profiles (Radar Chart)")
fig_radar = plot_radar_chart(cluster_profiles, features)
st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")

# =========================
# Cluster Statistics
# =========================
st.subheader("📈 Cluster Statistics")
selected_var = st.selectbox("Select variable to analyze", features)
stats_df = df_cancer.groupby('cluster')[selected_var].agg(['mean', 'std', 'count']).round(2)
stats_df.columns = ['Mean', 'Std Dev', 'Count']
st.dataframe(stats_df, use_container_width=True)

# =========================
# Feature Distributions by Cluster
# =========================
st.subheader("📊 Feature Distributions per Cluster")
selected_feature = st.selectbox("Select feature to visualize distribution", features, index=0)
fig_hist = plot_histogram(df_cancer, selected_feature, cluster_col='cluster')
st.plotly_chart(fig_hist, use_container_width=True)


