import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data, compute_mixed_correlation
from utils.styles import apply_page_style

st.set_page_config(
    page_title="Correlation & Association Analysis",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_page_style("Correlations")

# =========================
# Load Data
# =========================
df = load_data()

# =========================
# Hero Section
# =========================
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <div class="hero-title">🔄 Correlation & Association Analysis</div>
        <div class="hero-subtitle">
            Explore relationships between clinical variables through advanced statistical measures. 
            Identify patterns, dependencies, and predictive associations in lung cancer risk factors.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Control Panel
# =========================
st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.markdown('<div class="control-title">👥 Patient Group Selection</div>', unsafe_allow_html=True)

group_option = st.radio(
    "Analyze correlations for specific patient cohorts",
    ["All Patients", "Healthy Patients", "Lung Cancer Patients"],
    horizontal=True,
    help="Select a patient group to analyze variable associations within that cohort"
)

st.markdown('</div>', unsafe_allow_html=True)

# Filter data based on selection
if group_option == "Healthy Patients":
    df_group = df[df['lung_cancer'] == 0].copy()
    group_emoji = "✅"
    group_color = "#10b981"
elif group_option == "Lung Cancer Patients":
    df_group = df[df['lung_cancer'] == 1].copy()
    group_emoji = "🔴"
    group_color = "#dc2626"
else:
    df_group = df.copy()
    group_emoji = "👥"
    group_color = "#3b82f6"

# Info about selected group
st.markdown(f"""
<div class="info-box">
    <strong>{group_emoji} Selected Cohort:</strong> <strong style="color: {group_color};">{group_option}</strong>
    <br><br>
    Analyzing correlations across <strong>{len(df_group):,}</strong> patient records. 
    Associations are computed using mixed correlation methods suitable for both numerical and categorical variables.
</div>
""", unsafe_allow_html=True)

# =========================
# Quick Statistics
# =========================
st.markdown("""
<div style="margin: 40px 0 10px 0;">
    <div class="section-header">📊 Cohort Overview</div>
    <p class="section-subtitle">Key metrics for the selected patient group</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">👤</div>
        <div class="stat-label">Total Patients</div>
        <div class="stat-value">{len(df_group):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_age = df_group['age'].mean()
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">📅</div>
        <div class="stat-label">Average Age</div>
        <div class="stat-value">{avg_age:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    smoker_pct = (df_group['smoking'].sum() / len(df_group) * 100) if len(df_group) > 0 else 0
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">🚬</div>
        <div class="stat-label">Smokers</div>
        <div class="stat-value">{smoker_pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    num_vars = len(df_group.columns)
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">📋</div>
        <div class="stat-label">Variables</div>
        <div class="stat-value">{num_vars}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Mixed Correlation Matrix
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🔥 Association Heatmap</div>
    <p class="section-subtitle">Mixed correlation matrix showing relationships between all variables</p>
</div>
""", unsafe_allow_html=True)

# Legend for interpreting correlations
st.markdown("""
<div class="legend-box">
    <strong>📖 Interpretation Guide:</strong>
    <div class="strength-indicators">
        <span class="strength-badge badge-strong">
            <span style="font-size: 1.2rem;">●</span> Strong: |r| > 0.7
        </span>
        <span class="strength-badge badge-moderate">
            <span style="font-size: 1.2rem;">●</span> Moderate: 0.4 < |r| ≤ 0.7
        </span>
        <span class="strength-badge badge-weak">
            <span style="font-size: 1.2rem;">●</span> Weak: |r| ≤ 0.4
        </span>
    </div>
    <div style="margin-top: 12px;">
        <strong>Red colors</strong> indicate positive associations, <strong>blue colors</strong> indicate negative associations. 
        Values range from -1 (perfect negative) to +1 (perfect positive correlation).
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)

# Compute mixed correlations
numerical_cols = ['age']
assoc_matrix = compute_mixed_correlation(df_group, numerical_cols)

fig_heat = px.imshow(
    assoc_matrix,
    text_auto=".2f",
    color_continuous_scale='RdBu_r',
    aspect="auto",
    zmin=-1,
    zmax=1
)

fig_heat.update_layout(
    title=dict(
        text=f"<b>Variable Association Matrix - {group_option}</b>",
        font=dict(size=18, color="#1e293b")
    ),
    xaxis_title="Variables",
    yaxis_title="Variables",
    height=800,
    coloraxis_colorbar=dict(
        title=dict(
            text="Association<br>Strength",
            side="right"
        ),
        tickmode="linear",
        tick0=-1,
        dtick=0.5
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=11),
    margin=dict(t=60, b=80, l=100, r=40)
)

fig_heat.update_xaxes(tickangle=-45, side='bottom')
fig_heat.update_yaxes(tickangle=0)

st.plotly_chart(fig_heat, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Top Associations
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🔍 Strongest Associations</div>
    <p class="section-subtitle">Top 10 variable pairs ranked by association strength</p>
</div>
""", unsafe_allow_html=True)

# Flatten matrix to find top associations
assoc_long = assoc_matrix.where(np.triu(np.ones(assoc_matrix.shape), k=1).astype(bool))
assoc_long = assoc_long.stack().reset_index()
assoc_long.columns = ['Variable 1', 'Variable 2', 'Association']

# Sort by absolute association
assoc_long['Abs Association'] = assoc_long['Association'].abs()
assoc_long['Strength'] = assoc_long['Abs Association'].apply(
    lambda x: 'Strong' if x > 0.7 else ('Moderate' if x > 0.4 else 'Weak')
)
assoc_long['Direction'] = assoc_long['Association'].apply(
    lambda x: 'Positive ⬆' if x > 0 else 'Negative ⬇'
)

top_assoc = assoc_long.sort_values('Abs Association', ascending=False).head(10)
top_assoc = top_assoc[['Variable 1', 'Variable 2', 'Association', 'Abs Association', 'Strength', 'Direction']]
top_assoc['Rank'] = range(1, len(top_assoc) + 1)
top_assoc = top_assoc[['Rank', 'Variable 1', 'Variable 2', 'Association', 'Strength', 'Direction']]

st.markdown('<div class="data-table-container">', unsafe_allow_html=True)

st.dataframe(
    top_assoc.style.set_properties(**{
        'background-color': '#ffffff',
        'color': '#1e293b',
        'border-color': '#e2e8f0',
        'font-size': '14px',
        'font-family': 'Inter, sans-serif'
    }).format({'Association': '{:.3f}'}, precision=3),
    use_container_width=True,
    height=420
)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Pairwise Relationship Explorer
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">📈 Pairwise Relationship Explorer</div>
    <p class="section-subtitle">Interactive scatter matrix for selected variables</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.markdown('<div class="control-title">🎯 Variable Selection</div>', unsafe_allow_html=True)

# Get default variables (exclude lung_cancer from defaults if analyzing specific group)
default_vars = ['age', 'smoking', 'fatigue'] if 'fatigue' in df_group.columns else ['age', 'smoking']
if group_option == "All Patients":
    available_vars = [col for col in df_group.columns if col != 'lung_cancer']
else:
    available_vars = df_group.columns.tolist()

selected_vars = st.multiselect(
    "Select 2-4 variables to visualize pairwise relationships",
    options=available_vars,
    default=default_vars[:3],
    max_selections=4,
    help="Choose variables to create an interactive scatter matrix"
)

st.markdown('</div>', unsafe_allow_html=True)

if len(selected_vars) >= 2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Create scatter matrix
    if group_option == "All Patients" and 'lung_cancer' in df_group.columns:
        fig_pair = px.scatter_matrix(
            df_group,
            dimensions=selected_vars,
            color='lung_cancer',
            color_discrete_map={0: '#3b82f6', 1: '#dc2626'},
            labels={'lung_cancer': 'Lung Cancer Status'}
        )
    else:
        fig_pair = px.scatter_matrix(
            df_group,
            dimensions=selected_vars,
            color_discrete_sequence=['#dc2626']
        )
    
    fig_pair.update_traces(
        diagonal_visible=False,
        marker=dict(size=5, opacity=0.6, line=dict(width=0.5, color='white'))
    )
    
    fig_pair.update_layout(
        title=dict(
            text=f"<b>Pairwise Relationships - {group_option}</b>",
            font=dict(size=18, color="#1e293b")
        ),
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#334155", size=11),
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    st.plotly_chart(fig_pair, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Insight box
    st.markdown(f"""
    <div class="info-box">
        <strong>💡 Visualization Insight:</strong> The scatter matrix shows all pairwise relationships between 
        the {len(selected_vars)} selected variables. Each cell represents the relationship between two variables, 
        with points colored by {'cancer diagnosis status' if group_option == 'All Patients' else 'the selected cohort'}.
        Look for patterns, clusters, or trends that might indicate important clinical relationships.
    </div>
    """, unsafe_allow_html=True)
    
elif len(selected_vars) == 1:
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ Selection Required:</strong> Please select at least one more variable 
        to visualize pairwise relationships (currently {0} selected).
    </div>
    """.format(len(selected_vars)), unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ Selection Required:</strong> Please select at least 2 variables 
        to visualize pairwise relationships.
    </div>
    """, unsafe_allow_html=True)