import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data, compute_mixed_correlation

st.set_page_config(page_title="Correlations Analysis", page_icon="🔄", layout="wide")

# =========================
# Load data
# =========================
df = load_data()

# =========================
# Page title
# =========================
st.title("🔄 Correlation & Association Analysis")

# =========================
# Patient group selection
# =========================
group_option = st.radio(
    "Select patient group",
    ["All Patients", "Healthy Patients", "Lung Cancer Patients"]
)

if group_option == "Healthy Patients":
    df_group = df[df['lung_cancer'] == 0].copy()
elif group_option == "Lung Cancer Patients":
    df_group = df[df['lung_cancer'] == 1].copy()
else:
    df_group = df.copy()

# =========================
# Mixed correlation matrix
# =========================
st.subheader("📊 Mixed Correlation / Association Matrix")

# Compute mixed correlations (numerical + categorical)
numerical_cols = ['age']
assoc_matrix = compute_mixed_correlation(df_group, numerical_cols)

fig_heat = px.imshow(
    assoc_matrix,
    text_auto=".2f",
    color_continuous_scale='RdBu_r',
    aspect="auto",
    title=f"Variable Associations ({group_option})",
)
fig_heat.update_layout(
    xaxis_title="Variables",
    yaxis_title="Variables",
    height=800,
    width=900,
    coloraxis_colorbar=dict(title="Association Strength")
)
st.plotly_chart(fig_heat, use_container_width=True)

# =========================
# Show top associations
# =========================
st.subheader("🔍 Top Associations")

# Flatten matrix
assoc_long = assoc_matrix.where(np.triu(np.ones(assoc_matrix.shape), k=1).astype(bool))
assoc_long = assoc_long.stack().reset_index()
assoc_long.columns = ['Variable 1', 'Variable 2', 'Association']

# Sort by absolute association
assoc_long['abs_assoc'] = assoc_long['Association'].abs()
top_assoc = assoc_long.sort_values('abs_assoc', ascending=False).head(10)

st.dataframe(top_assoc[['Variable 1', 'Variable 2', 'Association']], use_container_width=True)

# =========================
# Optional: Interactive pair plots
# =========================
st.subheader("📈 Selected Variable Pair Plots")
selected_vars = st.multiselect(
    "Select up to 3 variables to visualize pairwise relationships",
    options=df_group.columns.tolist(),
    default=['age', 'smoking', 'fatigue'],
    max_selections=3
)

if len(selected_vars) >= 2:
    fig_pair = px.scatter_matrix(
        df_group,
        dimensions=selected_vars,
        color='lung_cancer' if 'lung_cancer' in df_group else None,
        title="Pairwise Relationships",
        color_discrete_map={0:'#636EFA', 1:'#EF553B'}
    )
    fig_pair.update_traces(diagonal_visible=False, marker=dict(size=6, opacity=0.7))
    st.plotly_chart(fig_pair, use_container_width=True)
else:
    st.info("Select at least 2 variables for pairwise visualization.")
