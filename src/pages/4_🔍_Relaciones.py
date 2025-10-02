import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.visualizations import plot_scatter_matrix

st.set_page_config(page_title="Relaciones entre Variables", page_icon="🔍")

# Cargar datos
df = load_data()

# Título
st.title("🔍 Relaciones entre Variables")

# Selector de variables
st.subheader("📊 Matriz de Dispersión")
variables = st.multiselect(
    "Seleccione variables para la matriz de dispersión",
    ['age'] + [col for col in df.columns if col not in ['gender', 'lung_cancer']],
    default=['age', 'smoking', 'anxiety', 'chronic_disease']
)

if len(variables) > 1:
    fig = plot_scatter_matrix(
        df,
        variables,
        color='lung_cancer'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Seleccione al menos dos variables para visualizar la matriz de dispersión")

# Scatter plot interactivo
st.subheader("📈 Gráfico de Dispersión Interactivo")

col1, col2 = st.columns(2)

with col1:
    x_var = st.selectbox(
        "Variable X",
        ['age'] + [col for col in df.columns if col not in ['gender', 'lung_cancer']],
        key='x_var'
    )

with col2:
    y_var = st.selectbox(
        "Variable Y",
        ['age'] + [col for col in df.columns if col not in ['gender', 'lung_cancer']],
        key='y_var'
    )

# Opciones adicionales
col1, col2 = st.columns(2)

with col1:
    color_var = st.selectbox(
        "Color por",
        ['lung_cancer', 'gender', 'None'],
        key='color_var'
    )

with col2:
    size_var = st.selectbox(
        "Tamaño por",
        ['None'] + ['age'] + [col for col in df.columns if col not in ['gender', 'lung_cancer']],
        key='size_var'
    )

# Crear scatter plot
fig = px.scatter(
    df,
    x=x_var,
    y=y_var,
    color=None if color_var == 'None' else color_var,
    size=None if size_var == 'None' else size_var,
    title=f'Relación entre {x_var} y {y_var}',
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)

# Estadísticas de las variables seleccionadas
st.subheader("📊 Estadísticas de las Variables Seleccionadas")

stats_df = df[[x_var, y_var]].describe()
st.dataframe(stats_df, use_container_width=True)

# Correlación entre las variables seleccionadas
correlation = df[x_var].corr(df[y_var])
st.metric(
    "Correlación",
    f"{correlation:.3f}"
)