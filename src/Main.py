import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import (
    load_data,
    compute_risk_score,
    get_statistics
)
from utils.visualizations import (
    create_kpi_cards,
    plot_risk_distribution,
    plot_feature_importance
)

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Lung Cancer Risk Analysis Platform",
    page_icon="🫁",
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
    
    .hero-section::after {
        content: "";
        position: absolute;
        bottom: -30%;
        left: -5%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.08) 0%, transparent 70%);
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
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin-bottom: 16px;
        line-height: 1.2;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: rgba(255, 255, 255, 0.95);
        line-height: 1.6;
        font-weight: 500;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 600;
        color: white;
        margin-top: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
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
    
    /* KPI Cards - Premium Design */
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
    
    @media (max-width: 768px) {
        .kpi-grid {
            grid-template-columns: 1fr;
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
    }
    
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
    }
    
    .kpi-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 32px rgba(220, 38, 38, 0.12);
        border-color: #fecaca;
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
        margin-bottom: 4px;
    }
    
    .kpi-trend {
        font-size: 13px;
        color: #059669;
        font-weight: 600;
    }
    
    /* Chart Containers */
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
    
    .chart-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 2px solid #f8fafc;
    }
    
    /* Data Table Styling */
    .data-table-container {
        background: white;
        padding: 32px;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
        margin-bottom: 40px;
    }
    
    /* Streamlit DataFrame Overrides */
    .stDataFrame {
        border: none !important;
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        border-radius: 12px;
        overflow: hidden;
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
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 20px 24px;
        border-radius: 12px;
        margin: 24px 0;
        font-size: 0.95rem;
        color: #92400e;
        line-height: 1.6;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid #e2e8f0;
        padding-top: 20px;
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        padding: 0 20px;
    }
    
    [data-testid="stSidebar"] h1 {
        color: #dc2626;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 2px solid #fee2e2;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* Footer */
    .custom-footer {
        background: white;
        border-top: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 40px 32px;
        margin-top: 60px;
        text-align: center;
        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.04);
    }
    
    .footer-content {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .footer-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 16px;
    }
    
    .footer-text {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 24px;
    }
    
    .footer-divider {
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
        margin: 24px auto;
        border-radius: 2px;
    }
    
    .social-section {
        margin-top: 28px;
    }
    
    .social-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 16px;
    }
    
    .social-links {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
    }
    
    .social-link {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 56px;
        height: 56px;
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-radius: 14px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-decoration: none;
        border: 2px solid transparent;
    }
    
    .social-link:hover {
        transform: translateY(-4px) scale(1.05);
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        border-color: #fca5a5;
        box-shadow: 0 8px 24px rgba(220, 38, 38, 0.25);
    }
    
    .social-link img {
        width: 28px;
        height: 28px;
        transition: all 0.3s ease;
        filter: brightness(0) saturate(100%) invert(21%) sepia(89%) saturate(3074%) hue-rotate(347deg) brightness(86%) contrast(92%);
    }
    
    .social-link:hover img {
        filter: brightness(0) saturate(100%) invert(100%);
    }
    
    .copyright {
        margin-top: 28px;
        padding-top: 24px;
        border-top: 1px solid #f1f5f9;
        color: #94a3b8;
        font-size: 0.85rem;
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
# Sidebar Filters
# =========================
with st.sidebar:
    st.title("🔬 Data Filters")
    
    st.markdown("### Patient Demographics")
    
    age_range = st.slider(
        "Age Range (years)",
        min_value=int(df['age'].min()),
        max_value=int(df['age'].max()),
        value=(int(df['age'].min()), int(df['age'].max()))
    )
    
    gender = st.multiselect(
        "Gender",
        options=df['gender'].unique(),
        default=df['gender'].unique()
    )
    
    smoking_map = {1: 'Smoker', 0: 'Non-Smoker'}
    smoking = st.multiselect(
        "Smoking Status",
        options=list(smoking_map.values()),
        default=list(smoking_map.values())
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='padding: 16px; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
    border-radius: 12px; font-size: 13px; color: #991b1b; line-height: 1.5;'>
    <strong>ℹ️ Filter Guide</strong><br>
    Adjust filters to analyze specific patient cohorts. 
    All visualizations update dynamically.
    </div>
    """, unsafe_allow_html=True)

# Apply Filters
mask = (
    (df['age'].between(age_range[0], age_range[1])) &
    (df['gender'].isin(gender)) &
    (df['smoking'].map(smoking_map).isin(smoking))
)
filtered_df = df[mask]

if filtered_df.empty:
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ No Data Available</strong><br>
        No patients match the selected filter criteria. Please adjust your filters to view the analysis.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# =========================
# Hero Section
# =========================
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <div class="hero-title">🫁 Lung Cancer Risk Analysis Platform</div>
        <div class="hero-subtitle">
            Comprehensive statistical analysis and predictive modeling for lung cancer risk assessment. 
            Explore clinical patterns, risk factors, and comparative patient profiles through interactive visualizations.
        </div>
        <div class="hero-badge">📊 Advanced Medical Data Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# KPI Section
# =========================
kpis = create_kpi_cards(filtered_df)

st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)

st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">📈</div>
        <div class="kpi-title">Cancer Rate (In This Dataset)</div>
        <div class="kpi-value">{kpis['cancer_pct']:.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">👥</div>
        <div class="kpi-title">Average Age (All)</div>
        <div class="kpi-value">{kpis['avg_age']:.1f} yrs</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">🔴</div>
        <div class="kpi-title">Average Age (Cancer)</div>
        <div class="kpi-value">{kpis['avg_age_cancer']:.1f} yrs</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">✅</div>
        <div class="kpi-title">Average Age (Healthy)</div>
        <div class="kpi-value">{kpis['avg_age_no_cancer']:.1f} yrs</div>
    </div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Methodology Section
# =========================
st.markdown("""
<div style="margin: 60px 0 30px 0;">
    <div class="section-header">🔬 Methodology & Data Science Approach</div>
    <p class="section-subtitle">Technical overview of the analytical framework and statistical methods employed</p>
</div>
""", unsafe_allow_html=True)

col_method1, col_method2 = st.columns(2, gap="large")

with col_method1:
    st.markdown("""
    <div class="chart-container">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
            <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                        border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px;">
                📊
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #1e293b;">Data Processing</div>
        </div>
        <div style="color: #475569; line-height: 1.7; font-size: 0.95rem;">
            <ul style="margin: 0; padding-left: 20px;">
                <li style="margin-bottom: 8px;"><strong>Feature Engineering:</strong> Transformation and normalization of clinical variables</li>
                <li style="margin-bottom: 8px;"><strong>Data Validation:</strong> Quality checks and outlier detection algorithms</li>
                <li style="margin-bottom: 8px;"><strong>Statistical Analysis:</strong> Descriptive statistics and correlation matrices</li>
                <li style="margin-bottom: 0;"><strong>Real-time Filtering:</strong> Dynamic data subsetting based on user-defined criteria</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_method2:
    st.markdown("""
    <div class="chart-container">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
            <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                        border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px;">
                🤖
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #1e293b;">Predictive Modeling</div>
        </div>
        <div style="color: #475569; line-height: 1.7; font-size: 0.95rem;">
            <ul style="margin: 0; padding-left: 20px;">
                <li style="margin-bottom: 8px;"><strong>Risk Scoring:</strong> Multi-variable risk assessment algorithms</li>
                <li style="margin-bottom: 8px;"><strong>Feature Importance:</strong> Identification of key predictive variables</li>
                <li style="margin-bottom: 8px;"><strong>Pattern Recognition:</strong> Machine learning-based classification models</li>
                <li style="margin-bottom: 0;"><strong>Visualization:</strong> Interactive Plotly charts for data exploration</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Clinical Notes Section
st.markdown("""
<div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
            border-left: 4px solid #f59e0b; padding: 24px 28px; border-radius: 14px; 
            margin: 30px 0; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);">
    <div style="display: flex; align-items: flex-start; gap: 16px;">
        <div style="font-size: 32px; line-height: 1;">⚕️</div>
        <div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #92400e; margin-bottom: 10px;">
                Clinical Note
            </div>
            <div style="color: #78350f; line-height: 1.6; font-size: 0.95rem;">
                This dashboard is designed for research and educational purposes. All risk assessments and statistical 
                analyses should be interpreted by qualified healthcare professionals. The presented data does not replace 
                clinical diagnosis or medical advice. Consult with oncologists and pulmonologists for individual patient care decisions.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Info Box
st.markdown("""
<div class="info-box">
    <strong>📊 Dataset Overview:</strong> This analysis includes <strong>{}</strong> patient records 
    filtered by your selected criteria. The data encompasses behavioral, clinical, and demographic variables 
    associated with lung cancer risk.
</div>
""".format(len(filtered_df)), unsafe_allow_html=True)

# =========================
# Key Insights Section
# =========================
st.markdown("""
<div style="margin: 50px 0 30px 0;">
    <div class="section-header">🔍 Key Clinical Insights</div>
    <p class="section-subtitle">Automated analysis of the most significant patterns in the current dataset</p>
</div>
""", unsafe_allow_html=True)

# Calculate insights
cancer_rate = (filtered_df['lung_cancer'].sum() / len(filtered_df)) * 100
smoker_cancer_rate = (filtered_df[filtered_df['smoking'] == 1]['lung_cancer'].sum() / 
                       len(filtered_df[filtered_df['smoking'] == 1]) * 100) if len(filtered_df[filtered_df['smoking'] == 1]) > 0 else 0
age_diff = kpis['avg_age_cancer'] - kpis['avg_age_no_cancer']

col_insight1, col_insight2, col_insight3 = st.columns(3, gap="large")

with col_insight1:
    st.markdown(f"""
    <div class="chart-container" style="min-height: 180px; display: flex; flex-direction: column; justify-content: center;">
        <div style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 12px;">🚬</div>
            <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 8px; font-weight: 600;">SMOKING IMPACT</div>
            <div style="font-size: 1.5rem; font-weight: 800; color: #dc2626; margin-bottom: 8px;">{smoker_cancer_rate:.1f}%</div>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.4;">
                Cancer rate among smokers in dataset
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_insight2:
    st.markdown(f"""
    <div class="chart-container" style="min-height: 180px; display: flex; flex-direction: column; justify-content: center;">
        <div style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 12px;">📅</div>
            <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 8px; font-weight: 600;">AGE DIFFERENCE</div>
            <div style="font-size: 1.5rem; font-weight: 800; color: #dc2626; margin-bottom: 8px;">+{age_diff:.1f} yrs</div>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.4;">
                Average age gap: Cancer vs. Healthy patients
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_insight3:
    gender_dist = filtered_df[filtered_df['lung_cancer'] == 1]['gender'].value_counts()
    most_affected = gender_dist.idxmax() if len(gender_dist) > 0 else "N/A"
    most_affected_pct = (gender_dist.max() / gender_dist.sum() * 100) if len(gender_dist) > 0 else 0
    
    st.markdown(f"""
    <div class="chart-container" style="min-height: 180px; display: flex; flex-direction: column; justify-content: center;">
        <div style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 12px;">👤</div>
            <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 8px; font-weight: 600;">GENDER DISTRIBUTION</div>
            <div style="font-size: 1.5rem; font-weight: 800; color: #dc2626; margin-bottom: 8px;">{most_affected}</div>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.4;">
                {most_affected_pct:.1f}% of cancer cases
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Risk Analysis Section
# =========================
risk_scores, feature_importance = compute_risk_score(filtered_df)

st.markdown("""
<div style="margin: 60px 0 0 0;">
    <div class="section-header">📊 Risk Distribution & Predictive Features</div>
    <p class="section-subtitle">Statistical analysis of computed risk scores and feature importance in lung cancer prediction</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Risk Score Distribution by Cancer Status</div>', unsafe_allow_html=True)
    fig1 = plot_risk_distribution(risk_scores, filtered_df['lung_cancer'])
    fig1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#334155", size=12),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Top Predictive Features</div>', unsafe_allow_html=True)
    fig2 = plot_feature_importance(feature_importance)
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#334155", size=12),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Comparative Statistics
# =========================
st.markdown("""
<div style="margin: 60px 0 0 0;">
    <div class="section-header">📈 Comparative Patient Statistics</div>
    <p class="section-subtitle">Side-by-side comparison of clinical and behavioral characteristics between cancer and non-cancer groups</p>
</div>
""", unsafe_allow_html=True)

group_col = 'lung_cancer'
stats = get_statistics(filtered_df, group_by=group_col)
all_groups = pd.Series([0, 1], name=group_col)
stats = stats.reindex(all_groups).fillna("—")
stats.index = stats.index.map({0: "Non-Cancer Patients", 1: "Lung Cancer Patients"})
stats.index.name = "Patient Group"

st.markdown('<div class="data-table-container">', unsafe_allow_html=True)
st.dataframe(
    stats.style.set_properties(**{
        'background-color': '#ffffff',
        'color': '#1e293b',
        'border-color': '#e2e8f0',
        'font-size': '14px',
        'font-family': 'Inter, sans-serif'
    }).format(precision=2),
    use_container_width=True,
    height=400
)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Custom Footer with Social Links
# =========================
st.markdown("""
<div class="custom-footer">
    <div class="footer-content">
        <div class="footer-title">Lung Cancer Risk Analysis Platform</div>
        <div class="footer-text">
            Advanced medical data analytics powered by machine learning and statistical modeling. 
            This platform provides healthcare professionals and researchers with actionable insights 
            for early detection and risk stratification.
        </div>
        <div class="footer-divider"></div>
        <div class="social-section">
            <div class="social-title">Connect With Me</div>
            <div class="social-links">
                <a href="https://github.com/yourusername" target="_blank" class="social-link" title="GitHub">
                    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub">
                </a>
                <a href="https://linkedin.com/in/yourusername" target="_blank" class="social-link" title="LinkedIn">
                    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn">
                </a>
            </div>
        </div>
        <div class="copyright">
            © 2025 Lung Cancer Analysis Dashboard | Built with Streamlit & Python | Data Science Research Project
        </div>
    </div>
</div>
""", unsafe_allow_html=True)