import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data, perform_clustering
from utils.visualizations import plot_cluster_3d, plot_radar_chart, plot_histogram

from utils.styles import apply_page_style

st.set_page_config(
    page_title="Lung Cancer Patient Clustering",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_page_style("Clusters")

# =========================
# Load Data
# =========================
df = load_data()

# Filter only patients with cancer
df_cancer = df[df['lung_cancer'] == 1].copy()

if df_cancer.empty:
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ No Data Available</strong><br>
        No lung cancer patients found in the dataset. Please check your data source.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# =========================
# Hero Section
# =========================
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <div class="hero-title">🎯 Lung Cancer Patient Clustering</div>
        <div class="hero-subtitle">
            Unsupervised machine learning analysis to identify distinct patient subgroups within lung cancer cases. 
            Discover patterns in symptoms, behaviors, and risk factors through advanced clustering techniques.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Clustering Configuration
# =========================
st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.markdown('<div class="control-title">⚙️ Clustering Configuration</div>', unsafe_allow_html=True)

col_slider, col_info = st.columns([2, 3])

with col_slider:
    n_clusters = st.slider(
        "Number of Patient Clusters",
        min_value=2,
        max_value=6,
        value=3,
        help="Select how many distinct patient groups to identify"
    )

with col_info:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); 
                padding: 16px 20px; border-radius: 12px; border-left: 4px solid #dc2626;">
        <div style="font-weight: 700; color: #991b1b; margin-bottom: 8px;">
            Selected Configuration: {n_clusters} Clusters
        </div>
        <div style="font-size: 0.9rem; color: #7f1d1d; line-height: 1.5;">
            The algorithm will segment {len(df_cancer):,} cancer patients into {n_clusters} distinct groups 
            based on {15} clinical and behavioral features.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Perform Clustering
# =========================
features = ['age_scaled', 'gender', 'smoking', 'yellow_fingers', 'anxiety', 'peer_pressure',
            'chronic_disease', 'fatigue', 'allergy', 'wheezing',
            'alcohol_consuming', 'coughing', 'shortness_of_breath',
            'swallowing_difficulty', 'chest_pain']

with st.spinner('Performing clustering analysis...'):
    clusters, X_pca, cluster_profiles = perform_clustering(df_cancer, features=features, n_clusters=n_clusters)
    df_cancer['cluster'] = clusters.astype(str)

# =========================
# Cluster Overview Statistics
# =========================
st.markdown("""
<div style="margin: 40px 0 10px 0;">
    <div class="section-header">📊 Cluster Distribution Overview</div>
    <p class="section-subtitle">Patient count and characteristics across identified clusters</p>
</div>
""", unsafe_allow_html=True)

# Calculate cluster statistics
cluster_counts = df_cancer['cluster'].value_counts().sort_index()
cluster_stats_list = []

for i in range(n_clusters):
    cluster_data = df_cancer[df_cancer['cluster'] == str(i)]
    cluster_stats_list.append({
        'count': len(cluster_data),
        'avg_age': cluster_data['age'].mean() if 'age' in cluster_data.columns else 0,
        'smoker_pct': (cluster_data['smoking'].sum() / len(cluster_data) * 100) if len(cluster_data) > 0 else 0
    })

# Display cluster cards
cols = st.columns(n_clusters)
for i, col in enumerate(cols):
    with col:
        stats = cluster_stats_list[i]
        percentage = (stats['count'] / len(df_cancer) * 100)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-label">Cluster {i}</div>
            <div class="stat-value">{stats['count']}</div>
            <div style="margin-top: 12px; font-size: 0.85rem; color: #64748b;">
                <strong>{percentage:.1f}%</strong> of patients<br>
                Avg Age: <strong>{stats['avg_age']:.1f}</strong><br>
                Smokers: <strong>{stats['smoker_pct']:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Info box about methodology
st.markdown("""
<div class="info-box">
    <strong>🔬 Methodology:</strong> Clusters are identified using <strong>K-Means algorithm</strong> 
    with PCA (Principal Component Analysis) for dimensionality reduction. Each cluster represents 
    a distinct patient profile based on clinical symptoms, behavioral factors, and demographic characteristics.
</div>
""", unsafe_allow_html=True)

# =========================
# 3D Cluster Visualization
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🔹 3D Cluster Visualization</div>
    <p class="section-subtitle">Interactive three-dimensional representation using Principal Component Analysis</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)
fig_3d = plot_cluster_3d(X_pca, clusters)
fig_3d.update_layout(
    title=dict(
        text=f"<b>Patient Clusters in 3D Space ({n_clusters} Clusters)</b>",
        font=dict(size=18, color="#1e293b")
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=11),
    margin=dict(t=60, b=40, l=40, r=40)
)
st.plotly_chart(fig_3d, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>💡 Interpretation:</strong> Each point represents a patient, and colors represent different clusters. 
    The three axes are principal components that capture the most variance in the data. Closer points indicate 
    similar patient profiles, while distinct clusters suggest different clinical presentations.
</div>
""", unsafe_allow_html=True)

# =========================
# Cluster Radar Profiles
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">📊 Cluster Profile Comparison</div>
    <p class="section-subtitle">Radar chart comparing average feature values across all clusters</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)
fig_radar = plot_radar_chart(cluster_profiles, features)
fig_radar.update_layout(
    title=dict(
        text=f"<b>Clinical Feature Profiles by Cluster</b>",
        font=dict(size=18, color="#1e293b")
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=11),
    margin=dict(t=60, b=40, l=80, r=80)
)
st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
    <strong>📖 Reading the Chart:</strong> Each colored line represents a cluster's average profile. 
    Features closer to the edge indicate higher prevalence or severity in that cluster. 
    This visualization helps identify which symptoms and risk factors define each patient subgroup.
</div>
""", unsafe_allow_html=True)

# =========================
# Cluster Statistics Table
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">📈 Statistical Analysis by Cluster</div>
    <p class="section-subtitle">Detailed descriptive statistics for selected variables across clusters</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.markdown('<div class="control-title">🔍 Select Variable for Analysis</div>', unsafe_allow_html=True)

selected_var = st.selectbox(
    "Choose a clinical or behavioral variable",
    features,
    help="Select a variable to view detailed statistics grouped by cluster"
)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="data-table-container">', unsafe_allow_html=True)

stats_df = df_cancer.groupby('cluster')[selected_var].agg(['mean', 'std', 'min', 'max', 'count']).round(3)
stats_df.columns = ['Mean', 'Std Dev', 'Min', 'Max', 'Count']
stats_df.index = [f'Cluster {i}' for i in stats_df.index]
stats_df.index.name = 'Patient Group'

st.dataframe(
    stats_df.style.set_properties(**{
        'background-color': '#ffffff',
        'color': '#1e293b',
        'border-color': '#e2e8f0',
        'font-size': '14px',
        'font-family': 'Inter, sans-serif'
    }).format(precision=3),
    use_container_width=True,
    height=300
)

st.markdown('</div>', unsafe_allow_html=True)

# Calculate which cluster has highest mean
max_cluster = stats_df['Mean'].idxmax()
max_value = stats_df['Mean'].max()

st.markdown(f"""
<div class="info-box">
    <strong>📊 Key Finding:</strong> <strong>{max_cluster}</strong> shows the highest average value 
    for <strong>{selected_var}</strong> ({max_value:.3f}), suggesting this feature is most prevalent 
    in this patient subgroup.
</div>
""", unsafe_allow_html=True)

# =========================
# Feature Distribution by Cluster
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">📊 Feature Distribution Analysis</div>
    <p class="section-subtitle">Histogram showing how selected features vary across clusters</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.markdown('<div class="control-title">🎨 Select Feature for Distribution Plot</div>', unsafe_allow_html=True)

selected_feature = st.selectbox(
    "Choose a feature to visualize its distribution",
    features,
    index=0,
    key="feature_dist",
    help="View how this feature is distributed within each cluster"
)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)
fig_hist = plot_histogram(df_cancer, selected_feature, cluster_col='cluster')
fig_hist.update_layout(
    title=dict(
        text=f"<b>{selected_feature.replace('_', ' ').title()} Distribution by Cluster</b>",
        font=dict(size=18, color="#1e293b")
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=12),
    margin=dict(t=60, b=40, l=40, r=40),
    showlegend=True,
    legend=dict(
        title="Cluster",
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)
st.plotly_chart(fig_hist, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Clinical Insights Summary
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🔬 Clinical Insights</div>
    <p class="section-subtitle">Key takeaways from the clustering analysis</p>
</div>
""", unsafe_allow_html=True)

col_insight1, col_insight2 = st.columns(2)

with col_insight1:
    st.markdown("""
    <div class="chart-container">
        <div style="text-align: center; margin-bottom: 16px;">
            <div style="font-size: 2.5rem;">🎯</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #1e293b; margin-bottom: 12px;">
                Patient Segmentation
            </div>
        </div>
        <div style="color: #475569; line-height: 1.7; font-size: 0.95rem;">
            The clustering algorithm has identified <strong>{0} distinct patient groups</strong> among 
            lung cancer cases. Each cluster represents patients with similar clinical presentations, 
            symptom patterns, and risk factor profiles. This segmentation can inform personalized 
            treatment strategies and targeted interventions.
        </div>
    </div>
    """.format(n_clusters), unsafe_allow_html=True)

with col_insight2:
    # Calculate cluster diversity (standard deviation of cluster sizes)
    cluster_sizes = [stats['count'] for stats in cluster_stats_list]
    size_std = np.std(cluster_sizes)
    balance_score = 100 - (size_std / np.mean(cluster_sizes) * 100)
    
    st.markdown(f"""
    <div class="chart-container">
        <div style="text-align: center; margin-bottom: 16px;">
            <div style="font-size: 2.5rem;">⚖️</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #1e293b; margin-bottom: 12px;">
                Cluster Balance
            </div>
        </div>
        <div style="color: #475569; line-height: 1.7; font-size: 0.95rem;">
            The clusters show a <strong>{balance_score:.1f}%</strong> balance score, indicating 
            {'relatively even distribution' if balance_score > 70 else 'varied distribution'} of patients 
            across groups. The largest cluster contains <strong>{max(cluster_sizes)}</strong> patients 
            while the smallest has <strong>{min(cluster_sizes)}</strong>, suggesting 
            {'diverse but balanced patient profiles' if balance_score > 70 else 'some distinct minority subgroups'}.
        </div>
    </div>
    """, unsafe_allow_html=True)

