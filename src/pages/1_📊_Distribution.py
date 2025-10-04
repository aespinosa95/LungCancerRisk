import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from utils.data_loader import load_data
from utils.styles import apply_page_style

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Variable Distribution Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Apply Styles and Navigation
# =========================
apply_page_style("Distribution")


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
    
    .chart-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 2px solid #f8fafc;
    }
    
    /* Stats Cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 24px 0;
    }
    
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        border: 1px solid #f1f5f9;
        border-top: 3px solid #dc2626;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(220, 38, 38, 0.12);
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #dc2626;
        line-height: 1.2;
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
    
    /* Variable Type Badge */
    .variable-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 12px;
    }
    
    .badge-numerical {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
    }
    
    .badge-categorical {
        background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%);
        color: #9f1239;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    /* Streamlit Overrides */
    .stSelectbox label, .stCheckbox label {
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
# Hero Section
# =========================
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <div class="hero-title">📊 Variable Distribution Analysis</div>
        <div class="hero-subtitle">
            Comprehensive statistical exploration of clinical and behavioral variables. 
            Analyze distributions, compare groups, and uncover patterns in lung cancer risk factors.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Control Panel
# =========================
st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.markdown('<div class="control-title">⚙️ Analysis Configuration</div>', unsafe_allow_html=True)

col_select, col_group = st.columns([3, 1])

with col_select:
    all_vars = ['age'] + [col for col in df.columns if col not in ['gender', 'lung_cancer', 'age']]
    variable = st.selectbox(
        "Select Variable to Analyze",
        all_vars,
        help="Choose a clinical or behavioral variable to explore its distribution"
    )

with col_group:
    group_by_cancer = st.checkbox(
        "Group by Cancer Diagnosis",
        help="Compare distributions between cancer and non-cancer patients"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Determine variable type
if df[variable].nunique() <= 6 or df[variable].dtype.name == 'category':
    var_type = 'categorical'
    type_icon = '📑'
    type_color = 'badge-categorical'
else:
    var_type = 'numerical'
    type_icon = '📈'
    type_color = 'badge-numerical'

# Variable Type Info
st.markdown(f"""
<div class="info-box">
    <strong>{type_icon} Variable Type:</strong> 
    <span class="variable-badge {type_color}">{var_type.upper()}</span>
    <br><br>
    Analyzing <strong>{variable}</strong> across <strong>{len(df):,}</strong> patient records. 
    {'Comparing distributions between lung cancer and non-cancer groups.' if group_by_cancer else 'Viewing overall distribution pattern.'}
</div>
""", unsafe_allow_html=True)

# =========================
# Quick Statistics Cards
# =========================
if var_type == 'numerical':
    st.markdown("""
    <div style="margin: 40px 0 10px 0;">
        <div class="section-header">📊 Quick Statistics</div>
        <p class="section-subtitle">Summary metrics for the selected variable</p>
    </div>
    """, unsafe_allow_html=True)
    
    if group_by_cancer:
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
            cancer_data = df[df['lung_cancer'] == 1][variable]
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">Cancer Group - Mean</div>
                    <div class="stat-value">{cancer_data.mean():.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Cancer Group - Median</div>
                    <div class="stat-value">{cancer_data.median():.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Cancer Group - Std Dev</div>
                    <div class="stat-value">{cancer_data.std():.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_stat2:
            st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
            no_cancer_data = df[df['lung_cancer'] == 0][variable]
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">Healthy Group - Mean</div>
                    <div class="stat-value">{no_cancer_data.mean():.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Healthy Group - Median</div>
                    <div class="stat-value">{no_cancer_data.median():.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Healthy Group - Std Dev</div>
                    <div class="stat-value">{no_cancer_data.std():.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Mean</div>
                <div class="stat-value">{df[variable].mean():.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Median</div>
                <div class="stat-value">{df[variable].median():.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Std Deviation</div>
                <div class="stat-value">{df[variable].std():.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Min Value</div>
                <div class="stat-value">{df[variable].min():.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max Value</div>
                <div class="stat-value">{df[variable].max():.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Visualization Section
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">📈 Distribution Visualization</div>
    <p class="section-subtitle">Interactive graphical representation of the data distribution</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)

if var_type == 'numerical':
    # Numerical: histogram + boxplot
    if group_by_cancer:
        fig = px.histogram(
            df,
            x=variable,
            color='lung_cancer',
            marginal='box',
            nbins=30,
            barmode='overlay',
            color_discrete_map={0: '#3b82f6', 1: '#dc2626'},
            labels={'lung_cancer': 'Lung Cancer Diagnosis'},
            opacity=0.7
        )
        fig.update_layout(
            title=dict(
                text=f"<b>{variable.replace('_', ' ').title()} Distribution by Cancer Diagnosis</b>",
                font=dict(size=18, color="#1e293b")
            ),
            legend=dict(
                title="Patient Group",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
    else:
        fig = px.histogram(
            df,
            x=variable,
            marginal='box',
            nbins=30,
            color_discrete_sequence=['#3b82f6']
        )
        fig.update_layout(
            title=dict(
                text=f"<b>{variable.replace('_', ' ').title()} Distribution</b>",
                font=dict(size=18, color="#1e293b")
            )
        )
    
    fig.update_xaxes(title=variable.replace('_', ' ').title(), showgrid=True, gridcolor='#f1f5f9')
    fig.update_yaxes(title="Frequency", showgrid=True, gridcolor='#f1f5f9')
    
else:
    # Categorical: bar chart
    if group_by_cancer:
        count_df = df.groupby([variable, 'lung_cancer']).size().reset_index(name='count')
        fig = px.bar(
            count_df,
            x=variable,
            y='count',
            color='lung_cancer',
            barmode='group',
            color_discrete_map={0: '#3b82f6', 1: '#dc2626'},
            labels={'lung_cancer': 'Lung Cancer Diagnosis', 'count': 'Patient Count'}
        )
        fig.update_layout(
            title=dict(
                text=f"<b>{variable.replace('_', ' ').title()} Distribution by Cancer Diagnosis</b>",
                font=dict(size=18, color="#1e293b")
            ),
            legend=dict(
                title="Patient Group",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
    else:
        count_df = df[variable].value_counts().reset_index()
        count_df.columns = [variable, 'count']
        fig = px.bar(
            count_df,
            x=variable,
            y='count',
            color=variable,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            title=dict(
                text=f"<b>{variable.replace('_', ' ').title()} Distribution</b>",
                font=dict(size=18, color="#1e293b")
            ),
            showlegend=False
        )
    
    fig.update_xaxes(title=variable.replace('_', ' ').title(), showgrid=False)
    fig.update_yaxes(title="Count", showgrid=True, gridcolor='#f1f5f9')

# Common layout updates
fig.update_layout(
    template='plotly_white',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=12),
    margin=dict(t=60, b=40, l=40, r=40),
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Descriptive Statistics Table
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">📋 Detailed Statistical Summary</div>
    <p class="section-subtitle">Comprehensive descriptive statistics and distribution metrics</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="data-table-container">', unsafe_allow_html=True)

if var_type == 'numerical':
    if group_by_cancer:
        stats = df.groupby('lung_cancer')[variable].describe()
        stats.index = stats.index.map({0: 'Healthy Patients', 1: 'Cancer Patients'})
        stats.index.name = 'Patient Group'
    else:
        stats = df[variable].describe().to_frame(name='Statistics')
else:
    if group_by_cancer:
        stats = df.groupby('lung_cancer')[variable].value_counts(normalize=True).unstack(fill_value=0)
        stats.index = stats.index.map({0: 'Healthy Patients', 1: 'Cancer Patients'})
        stats.index.name = 'Patient Group'
        stats = (stats * 100).round(2)  # Convert to percentages
    else:
        stats = df[variable].value_counts(normalize=True).to_frame(name='Proportion')
        stats['Percentage'] = (stats['Proportion'] * 100).round(2)
        stats['Count'] = df[variable].value_counts()

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
# Additional Insights
# =========================
if group_by_cancer and var_type == 'numerical':
    st.markdown("""
    <div style="margin: 40px 0 10px 0;">
        <div class="section-header">🔍 Comparative Analysis</div>
        <p class="section-subtitle">Statistical comparison between cancer and non-cancer groups</p>
    </div>
    """, unsafe_allow_html=True)
    
    cancer_mean = df[df['lung_cancer'] == 1][variable].mean()
    healthy_mean = df[df['lung_cancer'] == 0][variable].mean()
    difference = cancer_mean - healthy_mean
    percent_diff = (difference / healthy_mean * 100) if healthy_mean != 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="chart-container" style="text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">{'📈' if difference > 0 else '📉'}</div>
            <div class="stat-label">Mean Difference</div>
            <div class="stat-value">{abs(difference):.2f}</div>
            <div style="margin-top: 8px; font-size: 0.9rem; color: #64748b;">
                {'Higher' if difference > 0 else 'Lower'} in cancer group
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="chart-container" style="text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">📊</div>
            <div class="stat-label">Percentage Difference</div>
            <div class="stat-value">{abs(percent_diff):.1f}%</div>
            <div style="margin-top: 8px; font-size: 0.9rem; color: #64748b;">
                Relative change from baseline
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Calculate overlap coefficient
        from scipy import stats as scipy_stats
        cancer_std = df[df['lung_cancer'] == 1][variable].std()
        healthy_std = df[df['lung_cancer'] == 0][variable].std()
        overlap = min(cancer_std, healthy_std) / max(cancer_std, healthy_std) * 100
        
        st.markdown(f"""
        <div class="chart-container" style="text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">🔄</div>
            <div class="stat-label">Variability Ratio</div>
            <div class="stat-value">{overlap:.1f}%</div>
            <div style="margin-top: 8px; font-size: 0.9rem; color: #64748b;">
                Distribution overlap measure
            </div>
        </div>
        """, unsafe_allow_html=True)