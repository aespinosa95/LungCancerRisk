import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from utils.data_loader import load_data

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Variable Distributions", page_icon="📊", layout="wide")

# Load data
df = load_data()

# Page title
st.title("📊 Variable Distribution Analysis")

# Variable selector
all_vars = ['age'] + [col for col in df.columns if col not in ['gender', 'lung_cancer', 'age']]
variable = st.selectbox("Select a variable to analyze", all_vars)

# Grouping option
group_by_cancer = st.checkbox("Group by Lung Cancer Diagnosis")

# Determine variable type
if df[variable].nunique() <= 6 or df[variable].dtype.name == 'category':
    var_type = 'categorical'
else:
    var_type = 'numerical'

# =========================
# Plotting
# =========================
if var_type == 'numerical':
    # Numeric: histogram + boxplot
    if group_by_cancer:
        fig = px.histogram(
            df,
            x=variable,
            color='lung_cancer',
            marginal='box',
            nbins=30,
            barmode='overlay',
            color_discrete_map={0:'#636EFA', 1:'#EF553B'},
            labels={'lung_cancer': 'Lung Cancer'},
            title=f"Distribution of {variable} by Lung Cancer Diagnosis"
        )
    else:
        fig = px.histogram(
            df,
            x=variable,
            marginal='box',
            nbins=30,
            color_discrete_sequence=['#636EFA'],
            title=f"Distribution of {variable}"
        )
else:
    # Categorical: bar chart with counts or percentages
    if group_by_cancer:
        count_df = df.groupby([variable, 'lung_cancer']).size().reset_index(name='count')
        fig = px.bar(
            count_df,
            x=variable,
            y='count',
            color='lung_cancer',
            barmode='group',
            color_discrete_map={0:'#636EFA', 1:'#EF553B'},
            labels={'lung_cancer':'Lung Cancer', 'count':'Count'},
            title=f"{variable} Counts by Lung Cancer Diagnosis"
        )
    else:
        count_df = df[variable].value_counts().reset_index()
        count_df.columns = [variable, 'count']
        fig = px.bar(
            count_df,
            x=variable,
            y='count',
            color=variable,
            color_discrete_sequence=px.colors.qualitative.Set2,
            title=f"Counts of {variable}"
        )

fig.update_layout(template='plotly_white')
st.plotly_chart(fig, use_container_width=True)

# =========================
# Descriptive Statistics
# =========================
st.subheader("📈 Descriptive Statistics")
if var_type == 'numerical':
    if group_by_cancer:
        stats = df.groupby('lung_cancer')[variable].describe()
    else:
        stats = df[variable].describe()
else:
    if group_by_cancer:
        stats = df.groupby('lung_cancer')[variable].value_counts(normalize=True).unstack(fill_value=0)
    else:
        stats = df[variable].value_counts(normalize=True)

st.dataframe(stats, use_container_width=True)
