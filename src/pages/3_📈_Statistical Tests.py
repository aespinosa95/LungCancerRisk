import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, normaltest

from utils.data_loader import load_data
from utils.styles import apply_page_style

st.set_page_config(
    page_title="Statistical Hypothesis Testing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_page_style("Statistical Tests")

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
        <div class="hero-title">📈 Statistical Hypothesis Testing</div>
        <div class="hero-subtitle">
            Rigorous statistical inference to validate differences between cancer and healthy patients. 
            Test hypotheses about clinical variables using parametric and non-parametric methods.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Info about methodology
st.markdown("""
<div class="info-box">
    <strong>🔬 Statistical Methodology:</strong> This section performs hypothesis tests to determine if observed 
    differences between cancer and non-cancer groups are statistically significant or could have occurred by chance. 
    All tests use α = 0.05 significance level (95% confidence).
</div>
""", unsafe_allow_html=True)

# =========================
# Data Preparation
# =========================
df_cancer = df[df['lung_cancer'] == 1]
df_healthy = df[df['lung_cancer'] == 0]

# Separate variables by type
numerical_vars = ['age']
categorical_vars = [col for col in df.columns if col not in ['lung_cancer', 'age', 'age_scaled']]

# =========================
# Quick Summary Statistics
# =========================
st.markdown("""
<div style="margin: 40px 0 10px 0;">
    <div class="section-header">📊 Sample Overview</div>
    <p class="section-subtitle">Group sizes for statistical testing</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">👥</div>
        <div class="stat-label">Total Sample Size</div>
        <div class="stat-value">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">🔴</div>
        <div class="stat-label">Cancer Patients</div>
        <div class="stat-value">{len(df_cancer):,}</div>
        <div style="margin-top: 8px; font-size: 0.85rem; color: #64748b;">
            {len(df_cancer)/len(df)*100:.1f}% of sample
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">✅</div>
        <div class="stat-label">Healthy Patients</div>
        <div class="stat-value">{len(df_healthy):,}</div>
        <div style="margin-top: 8px; font-size: 0.85rem; color: #64748b;">
            {len(df_healthy)/len(df)*100:.1f}% of sample
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Age Analysis (Numerical)
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🔢 Age Comparison Analysis</div>
    <p class="section-subtitle">Testing if age differs significantly between cancer and healthy groups</p>
</div>
""", unsafe_allow_html=True)

# Normality tests
_, p_normal_cancer = normaltest(df_cancer['age'])
_, p_normal_healthy = normaltest(df_healthy['age'])

is_normal = (p_normal_cancer > 0.05) and (p_normal_healthy > 0.05)

# Choose appropriate test
if is_normal:
    # Independent t-test (parametric)
    t_stat, p_value_age = ttest_ind(df_cancer['age'], df_healthy['age'])
    test_name = "Independent Samples t-test"
    test_type = "Parametric"
else:
    # Mann-Whitney U test (non-parametric)
    u_stat, p_value_age = mannwhitneyu(df_cancer['age'], df_healthy['age'], alternative='two-sided')
    test_name = "Mann-Whitney U test"
    test_type = "Non-parametric"

# Calculate effect size (Cohen's d)
mean_cancer = df_cancer['age'].mean()
mean_healthy = df_healthy['age'].mean()
std_cancer = df_cancer['age'].std()
std_healthy = df_healthy['age'].std()
pooled_std = np.sqrt((std_cancer**2 + std_healthy**2) / 2)
cohens_d = (mean_cancer - mean_healthy) / pooled_std

# Interpret effect size
if abs(cohens_d) < 0.2:
    effect_interpretation = "Negligible"
elif abs(cohens_d) < 0.5:
    effect_interpretation = "Small"
elif abs(cohens_d) < 0.8:
    effect_interpretation = "Medium"
else:
    effect_interpretation = "Large"

# Display test results
col_test1, col_test2, col_test3 = st.columns(3)

with col_test1:
    st.markdown(f"""
    <div class="chart-container" style="text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 12px;">📊</div>
        <div class="stat-label">Statistical Test</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #1e293b; margin: 12px 0;">
            {test_name}
        </div>
        <div style="font-size: 0.85rem; color: #64748b;">
            {test_type} method
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_test2:
    significance = "Significant ✓" if p_value_age < 0.05 else "Not Significant ✗"
    color = "#10b981" if p_value_age < 0.05 else "#ef4444"
    st.markdown(f"""
    <div class="chart-container" style="text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 12px;">🎯</div>
        <div class="stat-label">P-Value</div>
        <div style="font-size: 1.8rem; font-weight: 800; color: {color}; margin: 12px 0;">
            {p_value_age:.4f}
        </div>
        <div style="font-size: 0.85rem; color: #64748b;">
            {significance}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_test3:
    st.markdown(f"""
    <div class="chart-container" style="text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 12px;">📏</div>
        <div class="stat-label">Effect Size (Cohen's d)</div>
        <div style="font-size: 1.8rem; font-weight: 800; color: #2563eb; margin: 12px 0;">
            {cohens_d:.3f}
        </div>
        <div style="font-size: 0.85rem; color: #64748b;">
            {effect_interpretation} effect
        </div>
    </div>
    """, unsafe_allow_html=True)

# Interpretation
st.markdown(f"""
<div class="{'info-box' if p_value_age < 0.05 else 'warning-box'}">
    <strong>🔍 Statistical Interpretation:</strong> 
    The {test_name} {'reveals a statistically significant difference' if p_value_age < 0.05 else 'does not show a statistically significant difference'} 
    in age between cancer patients (mean: {mean_cancer:.2f} years) and healthy individuals (mean: {mean_healthy:.2f} years). 
    P-value = {p_value_age:.4f} {'< 0.05' if p_value_age < 0.05 else '≥ 0.05'}, indicating 
    {'we can reject the null hypothesis' if p_value_age < 0.05 else 'we cannot reject the null hypothesis'}. 
    The effect size is {effect_interpretation.lower()} (Cohen's d = {cohens_d:.3f}).
</div>
""", unsafe_allow_html=True)

# Visualization
st.markdown('<div class="chart-container">', unsafe_allow_html=True)

fig_age = go.Figure()

fig_age.add_trace(go.Box(
    y=df_cancer['age'],
    name='Cancer Patients',
    marker_color='#dc2626',
    boxmean='sd'
))

fig_age.add_trace(go.Box(
    y=df_healthy['age'],
    name='Healthy Patients',
    marker_color='#3b82f6',
    boxmean='sd'
))

fig_age.update_layout(
    title=dict(
        text=f"<b>Age Distribution Comparison (p = {p_value_age:.4f})</b>",
        font=dict(size=18, color="#1e293b")
    ),
    yaxis_title="Age (years)",
    showlegend=True,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=12),
    margin=dict(t=60, b=40, l=40, r=40)
)

st.plotly_chart(fig_age, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Categorical Variables (Chi-Square Tests)
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🔄 Categorical Variables Association Tests</div>
    <p class="section-subtitle">Chi-square tests for independence between symptoms/behaviors and cancer diagnosis</p>
</div>
""", unsafe_allow_html=True)

# Perform chi-square tests for all categorical variables
chi_square_results = []

for var in categorical_vars:
    contingency_table = pd.crosstab(df[var], df['lung_cancer'])
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate Cramér's V (effect size for chi-square)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    
    # Odds ratio for 2x2 tables
    if contingency_table.shape == (2, 2):
        a, b = contingency_table.iloc[0, 0], contingency_table.iloc[0, 1]
        c, d = contingency_table.iloc[1, 0], contingency_table.iloc[1, 1]
        odds_ratio = (a * d) / (b * c) if (b * c) > 0 else np.nan
    else:
        odds_ratio = np.nan
    
    chi_square_results.append({
        'Variable': var,
        'Chi-Square': chi2,
        'P-Value': p_val,
        'Degrees of Freedom': dof,
        'Cramér\'s V': cramers_v,
        'Odds Ratio': odds_ratio,
        'Significant': 'Yes ✓' if p_val < 0.05 else 'No ✗'
    })

results_df = pd.DataFrame(chi_square_results).sort_values('P-Value')

# Summary stats
significant_vars = results_df[results_df['P-Value'] < 0.05]
st.markdown(f"""
<div class="info-box">
    <strong>📋 Summary:</strong> Out of {len(categorical_vars)} categorical variables tested, 
    <strong>{len(significant_vars)}</strong> show statistically significant associations with lung cancer 
    diagnosis (α = 0.05).
</div>
""", unsafe_allow_html=True)

# Display results table
st.markdown('<div class="data-table-container">', unsafe_allow_html=True)

display_df = results_df[['Variable', 'Chi-Square', 'P-Value', 'Cramér\'s V', 'Odds Ratio', 'Significant']]

st.dataframe(
    display_df.style.set_properties(**{
        'background-color': '#ffffff',
        'color': '#1e293b',
        'border-color': '#e2e8f0',
        'font-size': '14px',
        'font-family': 'Inter, sans-serif'
    }).format({
        'Chi-Square': '{:.3f}',
        'P-Value': '{:.4f}',
        'Cramér\'s V': '{:.3f}',
        'Odds Ratio': '{:.3f}'
    }, precision=3, na_rep='N/A'),
    use_container_width=True,
    height=500
)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Top Significant Associations Visualization
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🎯 Top Significant Risk Factors</div>
    <p class="section-subtitle">Variables with strongest statistical association to lung cancer</p>
</div>
""", unsafe_allow_html=True)

# Get top 8 most significant variables
top_vars = significant_vars.nsmallest(8, 'P-Value')

if len(top_vars) > 0:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Create bar chart of effect sizes
    fig_effect = go.Figure()
    
    fig_effect.add_trace(go.Bar(
        x=top_vars['Cramér\'s V'],
        y=top_vars['Variable'],
        orientation='h',
        marker=dict(
            color=top_vars['P-Value'],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="P-Value")
        ),
        text=[f"p={p:.4f}" for p in top_vars['P-Value']],
        textposition='auto',
    ))
    
    fig_effect.update_layout(
        title=dict(
            text="<b>Effect Sizes (Cramér's V) for Significant Associations</b>",
            font=dict(size=18, color="#1e293b")
        ),
        xaxis_title="Cramér's V (Effect Size)",
        yaxis_title="",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#334155", size=12),
        margin=dict(t=60, b=40, l=150, r=40),
        height=400
    )
    
    st.plotly_chart(fig_effect, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="warning-box">
        <strong>ℹ️ No Significant Associations:</strong> None of the categorical variables show 
        statistically significant associations with lung cancer at the α = 0.05 level.
    </div>
    """, unsafe_allow_html=True)

# =========================
# Interactive Variable Explorer
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🔬 Detailed Variable Analysis</div>
    <p class="section-subtitle">Explore contingency tables and proportions for individual variables</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.markdown('<div class="control-title">🎯 Select Variable for Detailed Analysis</div>', unsafe_allow_html=True)

selected_var = st.selectbox(
    "Choose a categorical variable",
    categorical_vars,
    help="View detailed statistical breakdown for this variable"
)

st.markdown('</div>', unsafe_allow_html=True)

# Get the chi-square result for selected variable
selected_result = results_df[results_df['Variable'] == selected_var].iloc[0]

# Create contingency table
cont_table = pd.crosstab(df[selected_var], df['lung_cancer'], margins=True, margins_name='Total')
cont_table.columns = ['Healthy', 'Cancer', 'Total']
cont_table.index = cont_table.index.map({0: 'No', 1: 'Yes', 'Total': 'Total'})

# Calculate proportions
prop_table = pd.crosstab(df[selected_var], df['lung_cancer'], normalize='columns') * 100
prop_table.columns = ['Healthy (%)', 'Cancer (%)']
prop_table.index = prop_table.index.map({0: 'No', 1: 'Yes'})

col_table1, col_table2 = st.columns(2)

with col_table1:
    st.markdown('<div class="data-table-container">', unsafe_allow_html=True)
    st.markdown("**Contingency Table (Counts)**")
    st.dataframe(
        cont_table.style.set_properties(**{
            'background-color': '#ffffff',
            'color': '#1e293b',
            'border-color': '#e2e8f0',
            'font-size': '14px',
            'font-family': 'Inter, sans-serif'
        }),
        use_container_width=True,
        height=180
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_table2:
    st.markdown('<div class="data-table-container">', unsafe_allow_html=True)
    st.markdown("**Proportion Table (%)**")
    st.dataframe(
        prop_table.style.set_properties(**{
            'background-color': '#ffffff',
            'color': '#1e293b',
            'border-color': '#e2e8f0',
            'font-size': '14px',
            'font-family': 'Inter, sans-serif'
        }).format(precision=1),
        use_container_width=True,
        height=180
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Statistical results for selected variable
st.markdown(f"""
<div class="{'info-box' if selected_result['P-Value'] < 0.05 else 'warning-box'}">
    <strong>🔍 Statistical Results for {selected_var.replace('_', ' ').title()}:</strong><br><br>
    • <strong>Chi-Square Statistic:</strong> {selected_result['Chi-Square']:.3f}<br>
    • <strong>P-Value:</strong> {selected_result['P-Value']:.4f} 
    {'(Statistically significant at α=0.05)' if selected_result['P-Value'] < 0.05 else '(Not statistically significant)'}<br>
    • <strong>Cramér\'s V (Effect Size):</strong> {selected_result['Cramér\'s V']:.3f}<br>
    {'• <strong>Odds Ratio:</strong> ' + f"{selected_result['Odds Ratio']:.3f}" if not pd.isna(selected_result['Odds Ratio']) else ''}<br>
    <br>
    <strong>Clinical Interpretation:</strong> 
    {'This variable shows a significant association with lung cancer diagnosis. ' if selected_result['P-Value'] < 0.05 else 'This variable does not show a significant association with lung cancer. '}
    {'The odds ratio indicates that presence of this factor ' + ('increases' if selected_result['Odds Ratio'] > 1 else 'decreases') + f" the odds of cancer by {abs(selected_result['Odds Ratio'] - 1)*100:.1f}%." if not pd.isna(selected_result['Odds Ratio']) and selected_result['P-Value'] < 0.05 else ''}
</div>
""", unsafe_allow_html=True)

# Visualization
st.markdown('<div class="chart-container">', unsafe_allow_html=True)

fig_var = px.bar(
    prop_table.reset_index(),
    x=selected_var,
    y=['Healthy (%)', 'Cancer (%)'],
    barmode='group',
    title=f"<b>{selected_var.replace('_', ' ').title()} Distribution by Cancer Status</b>",
    labels={selected_var: selected_var.replace('_', ' ').title(), 'value': 'Percentage (%)'},
    color_discrete_map={'Healthy (%)': '#3b82f6', 'Cancer (%)': '#dc2626'}
)

fig_var.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=12),
    margin=dict(t=60, b=40, l=40, r=40),
    legend=dict(title="Patient Group")
)

st.plotly_chart(fig_var, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Statistical Power Analysis
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">⚡ Statistical Power Considerations</div>
    <p class="section-subtitle">Understanding the reliability of our statistical conclusions</p>
</div>
""", unsafe_allow_html=True)

col_power1, col_power2 = st.columns(2)

with col_power1:
    st.markdown(f"""
    <div class="chart-container">
        <div style="text-align: center; margin-bottom: 16px;">
            <div style="font-size: 2.5rem;">📊</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #1e293b; margin-bottom: 12px;">
                Sample Size Adequacy
            </div>
        </div>
        <div style="color: #475569; line-height: 1.7; font-size: 0.95rem;">
            With <strong>{len(df):,}</strong> total observations ({len(df_cancer):,} cancer, {len(df_healthy):,} healthy), 
            our study has {'adequate' if min(len(df_cancer), len(df_healthy)) > 30 else 'limited'} statistical power 
            to detect medium-to-large effects. Smaller effect sizes may require larger sample sizes to achieve 
            statistical significance.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_power2:
    st.markdown("""
    <div class="chart-container">
        <div style="text-align: center; margin-bottom: 16px;">
            <div style="font-size: 2.5rem;">⚠️</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #1e293b; margin-bottom: 12px;">
                Multiple Testing Correction
            </div>
        </div>
        <div style="color: #475569; line-height: 1.7; font-size: 0.95rem;">
            When performing multiple statistical tests simultaneously, there is an increased risk of Type I errors 
            (false positives). Consider applying <strong>Bonferroni correction</strong> (adjusted α = 0.05/n) or 
            <strong>Benjamini-Hochberg FDR</strong> for more conservative inference.
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Methodology Notes
# =========================
st.markdown("""
<div class="warning-box">
    <strong>📖 Statistical Methods Used:</strong><br><br>
    <strong>1. Age Comparison:</strong><br>
    • <strong>Normality Test:</strong> D'Agostino-Pearson test to assess normal distribution<br>
    • <strong>Parametric:</strong> Independent samples t-test (if normal)<br>
    • <strong>Non-parametric:</strong> Mann-Whitney U test (if non-normal)<br>
    • <strong>Effect Size:</strong> Cohen's d (small: 0.2, medium: 0.5, large: 0.8)<br><br>
    
    <strong>2. Categorical Variables:</strong><br>
    • <strong>Association Test:</strong> Pearson's Chi-square test for independence<br>
    • <strong>Effect Size:</strong> Cramér's V (small: 0.1, medium: 0.3, large: 0.5)<br>
    • <strong>Odds Ratio:</strong> Risk measure for 2×2 contingency tables (OR > 1 indicates increased risk)<br><br>
    
    <strong>3. Significance Level:</strong> α = 0.05 (95% confidence)<br>
    <strong>4. Assumptions:</strong> Independence of observations, adequate expected cell counts (≥5) for chi-square
</div>
""", unsafe_allow_html=True)