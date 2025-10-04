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
from utils.styles import apply_page_style

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
# Apply Styles and Navigation
# =========================
apply_page_style("Home")

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
    <div style='padding: 16px; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
    border-radius: 12px; font-size: 13px; color: #1e40af; line-height: 1.5;'>
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
            <div style="font-size: 1.5rem; font-weight: 800; color: #2563eb; margin-bottom: 8px;">{smoker_cancer_rate:.1f}%</div>
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
            <div style="font-size: 1.5rem; font-weight: 800; color: #2563eb; margin-bottom: 8px;">+{age_diff:.1f} yrs</div>
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
            <div style="font-size: 1.5rem; font-weight: 800; color: #2563eb; margin-bottom: 8px;">{most_affected}</div>
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
            <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
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
            <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
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
                    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub" style="width: 28px; height: 28px;">
                </a>
                <a href="https://linkedin.com/in/yourusername" target="_blank" class="social-link" title="LinkedIn">
                    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" style="width: 28px; height: 28px;">
                </a>
            </div>
        </div>
        <div class="copyright">
            © 2025 Lung Cancer Analysis Dashboard | Built with Streamlit & Python | Data Science Research Project
        </div>
    </div>
</div>
""", unsafe_allow_html=True)