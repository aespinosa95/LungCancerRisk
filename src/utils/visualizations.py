import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

def create_kpi_cards(df):
    """
    Returns a dictionary of KPIs:
    - cancer_pct: prevalence of lung cancer in %
    - avg_age: average age overall
    - avg_age_cancer: average age of cancer patients
    - avg_age_no_cancer: average age of healthy patients
    """
    cancer_pct = df['lung_cancer'].mean() * 100  # 0/1 column
    avg_age = df['age'].mean()

    # Handle possible empty groups
    if (df['lung_cancer'] == 1).any():
        avg_age_cancer = df.loc[df['lung_cancer']==1, 'age'].mean()
    else:
        avg_age_cancer = np.nan

    if (df['lung_cancer'] == 0).any():
        avg_age_no_cancer = df.loc[df['lung_cancer']==0, 'age'].mean()
    else:
        avg_age_no_cancer = np.nan

    return {
        'cancer_pct': cancer_pct,
        'avg_age': avg_age,
        'avg_age_cancer': avg_age_cancer,
        'avg_age_no_cancer': avg_age_no_cancer
    }


def plot_risk_distribution(risk_scores, lung_cancer):
    """Plot risk score distribution."""
    fig = px.histogram(
        x=risk_scores,
        color=lung_cancer,
        nbins=30,
        title='Distribución de Puntuación de Riesgo por Diagnóstico',
        labels={'x': 'Puntuación de Riesgo', 'color': 'Cáncer de Pulmón'},
        template='plotly_white'
    )
    return fig

def plot_feature_importance(importance_df):
    """Plot feature importance chart."""
    colors = ['#ff4b4b' if d < 0 else '#1f77b4' for d in importance_df['direction']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color=colors
        )
    ])
    
    fig.update_layout(
        title='Importancia de Variables en la Predicción',
        xaxis_title='Importancia',
        yaxis_title='Variable',
        template='plotly_white'
    )
    
    return fig

def plot_correlation_heatmap(corr_matrix):
    """Plot correlation heatmap."""
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu',
        aspect='auto',
        title='Matriz de Correlación'
    )
    return fig

import plotly.express as px

def plot_cluster_3d(X_pca, clusters, show_axes_labels=True):
    """
    Create 3D scatter plot for clusters.

    Args:
        X_pca (np.array or DataFrame): PCA-transformed coordinates, shape (n_samples, 3)
        clusters (array-like): Cluster labels for each sample
        show_axes_labels (bool): Whether to display axis titles (default True)

    Returns:
        fig: Plotly 3D scatter figure
    """
    fig = px.scatter_3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=X_pca[:, 2],
        color=clusters.astype(str),
        title='3D Cluster Visualization',
        labels={'color': 'Cluster'},
        template='plotly_white'
    )

    if not show_axes_labels:
        fig.update_layout(scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title=''
        ))

    return fig


import plotly.graph_objects as go

def plot_radar_chart(cluster_profiles, features):
    """
    Plot radar chart for cluster profiles.

    Args:
        cluster_profiles (DataFrame): Each row is a cluster, columns are features.
        features (list): List of feature names to plot.
        
    Returns:
        fig: Plotly radar chart.
    """
    fig = go.Figure()

    for idx, row in cluster_profiles.iterrows():
        values = row[features].values.tolist()  # <-- Convertimos a lista de números
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=features,
            fill='toself',
            name=f'Cluster {idx}'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Opcional: normaliza valores entre 0 y 1
            )
        ),
        showlegend=True,
        template="plotly_white",
        title="Cluster Profiles Radar Chart"
    )
    return fig


def plot_variable_distribution(df, variable, hue=None):
    """Plot distribution of a variable with optional grouping."""
    if hue:
        fig = px.histogram(
            df,
            x=variable,
            color=hue,
            marginal='box',
            title=f'Distribución de {variable}',
            template='plotly_white'
        )
    else:
        fig = px.histogram(
            df,
            x=variable,
            title=f'Distribución de {variable}',
            template='plotly_white'
        )
    return fig

def plot_scatter_matrix(df, variables, color=None):
    """Create scatter matrix plot."""
    fig = px.scatter_matrix(
        df,
        dimensions=variables,
        color=color,
        title='Matriz de Dispersión',
        template='plotly_white'
    )
    return fig

import plotly.express as px

def plot_histogram(df, feature, cluster_col='cluster'):
    """
    Plot histogram of a feature separated by clusters.
    
    Args:
        df (DataFrame): Data containing feature and cluster columns.
        feature (str): Feature name to plot.
        cluster_col (str): Column indicating cluster assignment.
        
    Returns:
        fig: Plotly figure object.
    """
    fig = px.histogram(
        df,
        x=feature,
        color=cluster_col,
        barmode='overlay',
        marginal='box',  # Shows a mini boxplot on top
        nbins=20,
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Bold,
        title=f"{feature} Distribution by Cluster"
    )
    fig.update_layout(
        xaxis_title=feature,
        yaxis_title="Count",
        legend_title="Cluster",
        template="plotly_white",
        bargap=0.1,
        height=500
    )
    return fig
