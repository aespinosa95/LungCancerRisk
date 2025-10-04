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
    page_title="Lung Cancer Risk Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Advanced Medical Dashboard CSS
# =========================
st.markdown("""
<style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Main Background - Medical Theme */
    .main {
        background: linear-gradient(160deg, #f8fbff 0%, #f0f4f8 50%, #fef3f2 100%);
        background-attachment: fixed;
        color: #1e293b;
    }
    
    /* Subtle Medical Pattern Overlay */
    .main::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 15% 25%, rgba(220, 38, 38, 0.03) 0%, transparent 40%),
            radial-gradient(circle at 85% 75%, rgba(239, 68, 68, 0.02) 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(248, 113, 113, 0.015) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* Hero Header Section */
    .hero-section {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        padding: 48px 32px;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(220, 38, 38, 0.15);
        margin: 0 0 48px 0;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
        text-align: center;
        max-width: 1000px;
        margin: 0 auto;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin-bottom: 16px;
        line-height: 1.2;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .hero-subtitle {
        font-size: 1.15rem;
        color: rgba(255, 255, 255, 0.95);
        line-height: 1.6;
        font-weight: 500;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e293b;
        margin: 50px 0 28px 0;
        padding-bottom: 12px;
        border-bottom: 3px solid #fee2e2;
        display: flex;
        align-items: center;
        gap: 12px;
        position: relative;
    }
    
    .section-header::before {
        content: "";
        width: 8px;
        height: 28px;
        background: linear-gradient(180deg, #dc2626 0%, #b91c1c 100%);
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.3);
    }
    
    .section-subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-top: -20px;
        margin-bottom: 30px;
        font-weight: 500;
    }
    
    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin: 32px 0 48px 0;
    }
    
    @media (max-width: 1200px) {
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    .kpi-card {
        background: white;
        padding: 28px 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-top: 4px solid #dc2626;
    }
    
    .kpi-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 32px rgba(220, 38, 38, 0.12);
    }
    
    .kpi-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        margin-bottom: 16px;
    }
    
    .kpi-title {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #dc2626;
        line-height: 1.2;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        padding: 32px;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
        margin-bottom: 32px;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    }
    
    /* Control Panel */
    .control-panel {
        background: white;
        padding: 28px 32px;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
        margin-bottom: 40px;
        border-top: 4px solid #dc2626;
    }
    
    .control-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Data Table */
    .data-table-container {
        background: white;
        padding: 32px;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
        margin-bottom: 40px;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        padding: 20px 24px;
        border-radius: 12px;
        margin: 24px 0;
        font-size: 0.95rem;
        color: #1e40af;
        line-height: 1.6;
    }
    
    /* Warning Box */
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 20px 24px;
        border-radius: 12px;
        margin: 24px 0;
        font-size: 0.95rem;
        color: #78350f;
        line-height: 1.6;
    }
    
    /* Risk Level Badges */
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }
    
    .badge-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
    }
    
    .badge-medium {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #78350f;
    }
    
    .badge-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
    }
    
    /* Streamlit Overrides */
    .stSlider label {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================
# Load Data
# =========================
df = load_data()

# =========================
# Compute Risk Scores
# =========================
with st.spinner('Computing risk scores...'):
    risk_scores, feature_importance = compute_risk_score(df)
    df['risk_score'] = risk_scores

# Filter healthy patients
df_healthy = df[df['lung_cancer'] == 0].copy()

if df_healthy.empty:
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ No Data Available</strong><br>
        No healthy patients found in the dataset. Please check your data source.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# =========================
# Hero Section
# =========================
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <div class="hero-title">🧬 Lung Cancer Risk Prediction</div>
        <div class="hero-subtitle">
            Machine learning-powered risk assessment for healthy individuals. Analyze predicted cancer risk 
            based on clinical symptoms, behavioral factors, and demographic characteristics.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Info about cohort
st.markdown(f"""
<div class="info-box">
    <strong>👥 Analysis Cohort:</strong> This section focuses on <strong>{len(df_healthy):,} healthy patients</strong> 
    (currently without lung cancer diagnosis). Risk scores are computed using a predictive model trained on 
    clinical and behavioral features to estimate the likelihood of future lung cancer development.
</div>
""", unsafe_allow_html=True)

# =========================
# Risk Score KPIs
# =========================
st.markdown("""
<div style="margin: 40px 0 10px 0;">
    <div class="section-header">📊 Risk Score Distribution</div>
    <p class="section-subtitle">Statistical summary of predicted risk across healthy patients</p>
</div>
""", unsafe_allow_html=True)

avg_risk = df_healthy['risk_score'].mean()
median_risk = df_healthy['risk_score'].median()
max_risk = df_healthy['risk_score'].max()
min_risk = df_healthy['risk_score'].min()

st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">📊</div>
        <div class="kpi-title">Average Risk Score</div>
        <div class="kpi-value">{avg_risk:.3f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">📈</div>
        <div class="kpi-title">Median Risk Score</div>
        <div class="kpi-value">{median_risk:.3f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">⚠️</div>
        <div class="kpi-title">Maximum Risk</div>
        <div class="kpi-value">{max_risk:.3f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">✅</div>
        <div class="kpi-title">Minimum Risk</div>
        <div class="kpi-value">{min_risk:.3f}</div>
    </div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Risk categorization
low_risk = len(df_healthy[df_healthy['risk_score'] < 0.3])
med_risk = len(df_healthy[(df_healthy['risk_score'] >= 0.3) & (df_healthy['risk_score'] < 0.7)])
high_risk = len(df_healthy[df_healthy['risk_score'] >= 0.7])

st.markdown(f"""
<div class="warning-box">
    <strong>🎯 Risk Categorization:</strong> Among {len(df_healthy):,} healthy patients:<br>
    <div style="margin-top: 12px;">
        <span class="risk-badge badge-low">Low Risk (&lt;0.3): {low_risk} patients ({low_risk/len(df_healthy)*100:.1f}%)</span>
        <span class="risk-badge badge-medium">Medium Risk (0.3-0.7): {med_risk} patients ({med_risk/len(df_healthy)*100:.1f}%)</span>
        <span class="risk-badge badge-high">High Risk (≥0.7): {high_risk} patients ({high_risk/len(df_healthy)*100:.1f}%)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# 3D Scatter Plot
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🎯 3D Risk Visualization</div>
    <p class="section-subtitle">Interactive three-dimensional plot using the top 3 most important predictive features</p>
</div>
""", unsafe_allow_html=True)

top_features = feature_importance['feature'].head(3).tolist()

st.markdown('<div class="chart-container">', unsafe_allow_html=True)

fig_3d = px.scatter_3d(
    df_healthy,
    x=top_features[0],
    y=top_features[1],
    z=top_features[2],
    color='risk_score',
    size='risk_score',
    hover_data=['age'] + top_features,
    color_continuous_scale='Turbo',
    labels={'risk_score': 'Predicted Risk'}
)

fig_3d.update_layout(
    title=dict(
        text=f"<b>Risk Distribution in Feature Space</b>",
        font=dict(size=18, color="#1e293b")
    ),
    scene=dict(
        xaxis_title=top_features[0].replace('_',' ').title(),
        yaxis_title=top_features[1].replace('_',' ').title(),
        zaxis_title=top_features[2].replace('_',' ').title(),
        xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#f1f5f9"),
        yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#f1f5f9"),
        zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#f1f5f9")
    ),
    coloraxis_colorbar=dict(
        title=dict(text="Risk Score", side="right")
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=11),
    margin=dict(t=60, b=40, l=40, r=40)
)

st.plotly_chart(fig_3d, use_container_width=True, key="3d_risk_plot")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="info-box">
    <strong>💡 Interpretation:</strong> Each point represents a healthy patient positioned by their values in the 
    top 3 predictive features: <strong>{top_features[0].replace('_',' ').title()}</strong>, 
    <strong>{top_features[1].replace('_',' ').title()}</strong>, and 
    <strong>{top_features[2].replace('_',' ').title()}</strong>. 
    Color and size indicate predicted risk level.
</div>
""", unsafe_allow_html=True)

# =========================
# Heatmap by Age & Smoking
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">📊 Risk by Age & Smoking Status</div>
    <p class="section-subtitle">Heatmap showing average predicted risk across age groups and smoking behavior</p>
</div>
""", unsafe_allow_html=True)

age_bins = range(int(df_healthy['age'].min()//10*10), int(df_healthy['age'].max()//10*10 + 10), 10)
df_healthy['age_bin'] = pd.cut(df_healthy['age'], bins=age_bins).astype(str)
df_healthy['smoking_cat'] = df_healthy['smoking'].map({0:'Non-Smoker', 1:'Smoker'})

heatmap_data = df_healthy.pivot_table(
    index='age_bin',
    columns='smoking_cat',
    values='risk_score',
    aggfunc='mean'
)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)

fig_heat = px.imshow(
    heatmap_data,
    color_continuous_scale='Turbo',
    labels={'x':'Smoking Status', 'y':'Age Range', 'color':'Risk Score'},
    text_auto='.3f'
)

fig_heat.update_layout(
    title=dict(
        text="<b>Average Risk by Age Range & Smoking Status</b>",
        font=dict(size=18, color="#1e293b")
    ),
    yaxis={'autorange':'reversed'},
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=12),
    margin=dict(t=60, b=40, l=80, r=40)
)

st.plotly_chart(fig_heat, use_container_width=True, key="heatmap_smoking")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Violin Plots
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">📊 Risk Distribution by Clinical Features</div>
    <p class="section-subtitle">Violin plots showing how risk scores vary with presence/absence of key symptoms</p>
</div>
""", unsafe_allow_html=True)

categorical_relevant = ['fatigue', 'allergy', 'chronic_disease', 'coughing',
                        'swallowing_difficulty', 'alcohol_consuming']

df_healthy_v = df_healthy.copy()
for col in categorical_relevant:
    df_healthy_v[col + '_cat'] = df_healthy_v[col].map({0:'No', 1:'Yes'})

st.markdown('<div class="chart-container">', unsafe_allow_html=True)

fig_violin = make_subplots(
    rows=len(categorical_relevant), cols=1, shared_xaxes=False,
    subplot_titles=[f"{col.replace('_',' ').title()}" for col in categorical_relevant],
    vertical_spacing=0.08
)

colors = ['#3b82f6', '#dc2626', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']

for i, col in enumerate(categorical_relevant, start=1):
    fig_violin.add_trace(
        go.Violin(
            y=df_healthy_v['risk_score'],
            x=df_healthy_v[col + '_cat'],
            box_visible=True,
            meanline_visible=True,
            points='all',
            marker=dict(opacity=0.5, size=4),
            line_color=colors[i-1],
            fillcolor=colors[i-1],
            opacity=0.6,
            name=col.replace('_',' ').title()
        ),
        row=i, col=1
    )

fig_violin.update_layout(
    height=280*len(categorical_relevant),
    title_text="<b>Risk Score Distribution by Symptom Presence</b>",
    showlegend=False,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=11),
    margin=dict(t=60, b=40, l=40, r=40)
)

fig_violin.update_yaxes(title_text="Predicted Risk Score", showgrid=True, gridcolor='#f1f5f9')
fig_violin.update_xaxes(showgrid=False)

st.plotly_chart(fig_violin, use_container_width=True, key="violin_risk")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Feature Importance
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">📈 Feature Importance Analysis</div>
    <p class="section-subtitle">Ranking of variables by their contribution to risk prediction</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="data-table-container">', unsafe_allow_html=True)

# Add rank column
feature_importance_display = feature_importance.copy()
feature_importance_display['Rank'] = range(1, len(feature_importance_display) + 1)
feature_importance_display = feature_importance_display[['Rank', 'feature', 'importance']]
feature_importance_display.columns = ['Rank', 'Feature', 'Importance Score']

st.dataframe(
    feature_importance_display.style.set_properties(**{
        'background-color': '#ffffff',
        'color': '#1e293b',
        'border-color': '#e2e8f0',
        'font-size': '14px',
        'font-family': 'Inter, sans-serif'
    }).format({'Importance Score': '{:.4f}'}, precision=4),
    use_container_width=True,
    height=400
)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="info-box">
    <strong>🔑 Key Predictors:</strong> The top 3 most important features for risk prediction are:
    <strong>{feature_importance['feature'].iloc[0].replace('_',' ').title()}</strong> (importance: {feature_importance['importance'].iloc[0]:.4f}),
    <strong>{feature_importance['feature'].iloc[1].replace('_',' ').title()}</strong> (importance: {feature_importance['importance'].iloc[1]:.4f}), and
    <strong>{feature_importance['feature'].iloc[2].replace('_',' ').title()}</strong> (importance: {feature_importance['importance'].iloc[2]:.4f}).
</div>
""", unsafe_allow_html=True)

# =========================
# Interactive Filtering
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🔎 High-Risk Patient Explorer</div>
    <p class="section-subtitle">Filter and identify patients above a specified risk threshold</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.markdown('<div class="control-title">⚙️ Risk Threshold Configuration</div>', unsafe_allow_html=True)

risk_threshold = st.slider(
    "Select Minimum Risk Score",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Filter patients with predicted risk at or above this threshold"
)

st.markdown('</div>', unsafe_allow_html=True)

filtered_patients = df_healthy[df_healthy['risk_score'] >= risk_threshold]

st.markdown(f"""
<div class="info-box">
    <strong>📋 Filtered Results:</strong> <strong>{len(filtered_patients):,} patients</strong> 
    ({len(filtered_patients)/len(df_healthy)*100:.1f}% of healthy cohort) have a predicted risk score 
    ≥ {risk_threshold:.2f}. These individuals may benefit from enhanced screening or preventive interventions.
</div>
""", unsafe_allow_html=True)

if len(filtered_patients) > 0:
    st.markdown('<div class="data-table-container">', unsafe_allow_html=True)
    
    display_cols = ['age'] + categorical_relevant + ['risk_score']
    filtered_display = filtered_patients[display_cols].copy()
    filtered_display.columns = [col.replace('_', ' ').title() for col in display_cols]
    
    st.dataframe(
        filtered_display.style.set_properties(**{
            'background-color': '#ffffff',
            'color': '#1e293b',
            'border-color': '#e2e8f0',
            'font-size': '14px',
            'font-family': 'Inter, sans-serif'
        }).format({'Risk Score': '{:.4f}'}, precision=4),
        use_container_width=True,
        height=400
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="warning-box">
        <strong>ℹ️ No Matches:</strong> No patients meet the selected risk threshold criteria. 
        Try lowering the threshold to view results.
    </div>
    """, unsafe_allow_html=True)