import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import load_data, compute_risk_score

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Prediction of Lung Cancer Risk",
    page_icon="🧬",
    layout="wide"
)

# =========================
# Load Data
# =========================
df = load_data()

# =========================
# Page Title
# =========================
st.title("🧬 Lung Cancer Risk Prediction for Healthy Patients")

# =========================
# Compute Risk Scores
# =========================
# Compute risk scores over all patients (like in EDA)
risk_scores, feature_importance = compute_risk_score(df)
df['risk_score'] = risk_scores

# Filter healthy patients
df_healthy = df[df['lung_cancer'] == 0].copy()
st.subheader(f"Healthy Patients: {len(df_healthy)}")

if df_healthy.empty:
    st.warning("No healthy patients found in the dataset.")
    st.stop()


# =========================
# KPIs
# =========================
avg_risk = df_healthy['risk_score'].mean()
median_risk = df_healthy['risk_score'].median()
max_risk = df_healthy['risk_score'].max()
min_risk = df_healthy['risk_score'].min()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Predicted Risk", f"{avg_risk:.3f}")
col2.metric("Median Predicted Risk", f"{median_risk:.3f}")
col3.metric("Maximum Risk", f"{max_risk:.3f}")
col4.metric("Minimum Risk", f"{min_risk:.3f}")

# =========================
# 3D Scatter Plot (Top 3 Features)
# =========================
st.subheader("🎯 3D Risk Visualization (Top 3 Features)")

# Automatically select top 3 features by importance
top_features = feature_importance['feature'].head(3).tolist()

fig_3d = px.scatter_3d(
    df_healthy,
    x=top_features[0],
    y=top_features[1],
    z=top_features[2],
    color='risk_score',
    size='risk_score',
    hover_data=['age'] + top_features,
    color_continuous_scale='Turbo',
    title="Predicted Lung Cancer Risk (Top 3 Features)"
)
fig_3d.update_layout(
    scene=dict(
        xaxis_title=top_features[0].replace('_',' ').capitalize(),
        yaxis_title=top_features[1].replace('_',' ').capitalize(),
        zaxis_title=top_features[2].replace('_',' ').capitalize()
    ),
    coloraxis_colorbar=dict(title="Risk Score")
)
st.plotly_chart(fig_3d, use_container_width=True, key="3d_risk_plot")

# =========================
# Heatmap by Age & Smoking
# =========================
st.subheader("📊 Average Risk by Age Range & Smoking")

age_bins = range(int(df_healthy['age'].min()//10*10), int(df_healthy['age'].max()//10*10 + 10), 10)
df_healthy['age_bin'] = pd.cut(df_healthy['age'], bins=age_bins).astype(str)
df_healthy['smoking_cat'] = df_healthy['smoking'].map({0:'No', 1:'Yes'})

heatmap_data = df_healthy.pivot_table(
    index='age_bin',
    columns='smoking_cat',
    values='risk_score',
    aggfunc='mean'
)

fig_heat = px.imshow(
    heatmap_data,
    color_continuous_scale='Turbo',
    labels={'x':'Smoking', 'y':'Age Range', 'color':'Risk Score'},
    title="Average Predicted Risk by Age & Smoking"
)
fig_heat.update_layout(yaxis={'autorange':'reversed'})
st.plotly_chart(fig_heat, use_container_width=True, key="heatmap_smoking")

# =========================
# Violin Plots for Categorical Features
# =========================
st.subheader("📊 Risk by Top Categorical Variables")

categorical_relevant = ['fatigue', 'allergy', 'chronic_disease', 'coughing',
                        'swallowing_difficulty', 'alcohol_consuming']

# Convert 0/1 to Yes/No for display
df_healthy_v = df_healthy.copy()
for col in categorical_relevant:
    df_healthy_v[col + '_cat'] = df_healthy_v[col].map({0:'No', 1:'Yes'})

fig_violin = make_subplots(
    rows=len(categorical_relevant), cols=1, shared_xaxes=False,
    subplot_titles=[f"Risk by {col.replace('_',' ').capitalize()}" for col in categorical_relevant]
)

for i, col in enumerate(categorical_relevant, start=1):
    fig_violin.add_trace(
        go.Violin(
            y=df_healthy_v['risk_score'],
            x=df_healthy_v[col + '_cat'],
            box_visible=True,
            meanline_visible=True,
            points='all',
            marker=dict(opacity=0.6),
            name=col.replace('_',' ').capitalize()
        ),
        row=i, col=1
    )

fig_violin.update_layout(
    height=300*len(categorical_relevant),
    title_text="Predicted Risk by Top Categorical Features",
    showlegend=False
)
fig_violin.update_yaxes(title_text="Predicted Risk Score")
st.plotly_chart(fig_violin, use_container_width=True, key="violin_risk")

# =========================
# Feature Importance
# =========================
st.subheader("📊 Feature Importance for Risk Prediction")

st.dataframe(feature_importance.round(3), use_container_width=True)

# =========================
# Optional: Interactive Filtering
# =========================
st.subheader("🔎 Filter Patients by Predicted Risk")

risk_threshold = st.slider("Select minimum risk score", 0.0, 1.0, 0.1, 0.01)
filtered_patients = df_healthy[df_healthy['risk_score'] >= risk_threshold]

st.dataframe(filtered_patients[['age'] + categorical_relevant + ['risk_score']], use_container_width=True)
st.markdown(f"**Total Patients with Risk ≥ {risk_threshold:.2f}: {len(filtered_patients)}**")