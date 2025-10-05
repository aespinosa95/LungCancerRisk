import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_data
from utils.styles import apply_page_style

st.set_page_config(
    page_title="Model Performance Evaluation",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_page_style("Model Performance")

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
        <div class="hero-title">🎓 Model Performance Evaluation</div>
        <div class="hero-subtitle">
            Comprehensive assessment of machine learning models for lung cancer prediction. 
            Compare algorithms, validate performance, and evaluate generalization through rigorous metrics.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Data Preparation
# =========================
# Features (exclude target and identifiers)
feature_cols = [col for col in df.columns if col not in ['lung_cancer', 'age', 'age_scaled']]
X = df[feature_cols]
y = df['lung_cancer']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scaling for models that benefit from it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.markdown(f"""
<div class="info-box">
    <strong>📊 Dataset Split:</strong> Using <strong>75%</strong> of data for training ({len(X_train):,} samples) 
    and <strong>25%</strong> for testing ({len(X_test):,} samples). Stratified sampling ensures balanced 
    class distribution across splits.
</div>
""", unsafe_allow_html=True)

# =========================
# Model Selection Panel
# =========================
st.markdown("""
<div style="margin: 40px 0 10px 0;">
    <div class="section-header">🤖 Model Configuration</div>
    <p class="section-subtitle">Select machine learning algorithms to train and evaluate</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.markdown('<div class="control-title">⚙️ Algorithm Selection</div>', unsafe_allow_html=True)

col_model1, col_model2 = st.columns(2)

with col_model1:
    selected_models = st.multiselect(
        "Choose models to compare",
        ["Logistic Regression", "Random Forest", "Gradient Boosting", "Support Vector Machine"],
        default=["Logistic Regression", "Random Forest", "Gradient Boosting"],
        help="Select one or more classification algorithms"
    )

with col_model2:
    cv_folds = st.slider(
        "Cross-Validation Folds",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of folds for cross-validation (higher = more robust but slower)"
    )

st.markdown('</div>', unsafe_allow_html=True)

if len(selected_models) == 0:
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Selection Required:</strong> Please select at least one model to evaluate.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# =========================
# Train Models
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🔧 Model Training</div>
    <p class="section-subtitle">Training selected algorithms on the dataset</p>
</div>
""", unsafe_allow_html=True)

models_dict = {}
results = []

with st.spinner('Training models...'):
    
    # Define models
    if "Logistic Regression" in selected_models:
        models_dict["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=42)
    
    if "Random Forest" in selected_models:
        models_dict["Random Forest"] = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
    
    if "Gradient Boosting" in selected_models:
        models_dict["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
    
    if "Support Vector Machine" in selected_models:
        models_dict["Support Vector Machine"] = SVC(
            kernel='rbf', probability=True, random_state=42
        )
    
    # Train and evaluate each model
    for model_name, model in models_dict.items():
        # Use scaled data for SVM and Logistic Regression
        if model_name in ["Support Vector Machine", "Logistic Regression"]:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train.values, X_test.values
        
        # Train
        model.fit(X_tr, y_train)
        
        # Predictions
        y_pred = model.predict(X_te)
        y_pred_proba = model.predict_proba(X_te)[:, 1]
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='roc_auc')
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'CV Mean AUC': cv_scores.mean(),
            'CV Std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        })

results_df = pd.DataFrame(results)

st.success(f"✓ Successfully trained {len(models_dict)} models")

# =========================
# Performance Metrics Table
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">📊 Performance Metrics Comparison</div>
    <p class="section-subtitle">Comprehensive evaluation across multiple performance indicators</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="data-table-container">', unsafe_allow_html=True)

display_metrics = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'CV Mean AUC', 'CV Std']]

st.dataframe(
    display_metrics.style.set_properties(**{
        'background-color': '#ffffff',
        'color': '#1e293b',
        'border-color': '#e2e8f0',
        'font-size': '14px',
        'font-family': 'Inter, sans-serif'
    }).format({
        'Accuracy': '{:.3f}',
        'Precision': '{:.3f}',
        'Recall': '{:.3f}',
        'F1-Score': '{:.3f}',
        'ROC-AUC': '{:.3f}',
        'CV Mean AUC': '{:.3f}',
        'CV Std': '{:.4f}'
    }, precision=3).background_gradient(
        subset=['ROC-AUC', 'F1-Score', 'CV Mean AUC'],
        cmap='RdYlGn',
        vmin=0.5,
        vmax=1.0
    ),
    use_container_width=True,
    height=250
)

st.markdown('</div>', unsafe_allow_html=True)

# Best model identification
best_model_idx = results_df['ROC-AUC'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_auc = results_df.loc[best_model_idx, 'ROC-AUC']

st.markdown(f"""
<div class="info-box">
    <strong>🏆 Best Performing Model:</strong> <strong>{best_model_name}</strong> achieves the highest 
    ROC-AUC score of <strong>{best_auc:.3f}</strong> on the test set, with cross-validation AUC of 
    <strong>{results_df.loc[best_model_idx, 'CV Mean AUC']:.3f} ± {results_df.loc[best_model_idx, 'CV Std']:.3f}</strong>.
</div>
""", unsafe_allow_html=True)

# =========================
# Performance Metrics Visualization
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">📈 Visual Performance Comparison</div>
    <p class="section-subtitle">Radar chart and bar plots showing model strengths across metrics</p>
</div>
""", unsafe_allow_html=True)

col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Radar chart
    metrics_for_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    fig_radar = go.Figure()
    
    for idx, row in results_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics_for_radar],
            theta=metrics_for_radar,
            fill='toself',
            name=row['Model']
        ))
    
    fig_radar.update_layout(
        title=dict(
            text="<b>Multi-Metric Performance Radar</b>",
            font=dict(size=16, color="#1e293b")
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#334155", size=11),
        margin=dict(t=60, b=40, l=40, r=40),
        height=400
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_viz2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Bar chart comparison
    fig_bars = go.Figure()
    
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for metric in metrics_list:
        fig_bars.add_trace(go.Bar(
            name=metric,
            x=results_df['Model'],
            y=results_df[metric],
            text=results_df[metric].round(3),
            textposition='auto',
        ))
    
    fig_bars.update_layout(
        title=dict(
            text="<b>Performance Metrics by Model</b>",
            font=dict(size=16, color="#1e293b")
        ),
        barmode='group',
        yaxis_title="Score",
        xaxis_title="Model",
        yaxis=dict(range=[0, 1.05]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#334155", size=11),
        margin=dict(t=60, b=40, l=40, r=40),
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig_bars, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ROC Curves
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">📉 ROC Curves Analysis</div>
    <p class="section-subtitle">Receiver Operating Characteristic curves showing true positive vs false positive rates</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)

fig_roc = go.Figure()

# Plot ROC curve for each model
for idx, row in results_df.iterrows():
    fpr, tpr, _ = roc_curve(y_test, row['y_pred_proba'])
    roc_auc = auc(fpr, tpr)
    
    fig_roc.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f"{row['Model']} (AUC = {roc_auc:.3f})",
        line=dict(width=3)
    ))

# Add diagonal reference line
fig_roc.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Random Classifier',
    line=dict(dash='dash', color='gray', width=2)
))

fig_roc.update_layout(
    title=dict(
        text="<b>ROC Curves - All Models</b>",
        font=dict(size=18, color="#1e293b")
    ),
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=12),
    margin=dict(t=60, b=40, l=40, r=40),
    height=500,
    showlegend=True,
    legend=dict(x=0.6, y=0.1)
)

st.plotly_chart(fig_roc, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>💡 ROC Curve Interpretation:</strong> The ROC curve plots the true positive rate (sensitivity) 
    against the false positive rate (1-specificity) at various threshold settings. A perfect classifier 
    would have a curve that passes through the top-left corner (AUC = 1.0). The closer the curve follows 
    the diagonal line, the less accurate the model (AUC = 0.5 is random).
</div>
""", unsafe_allow_html=True)

# =========================
# Precision-Recall Curves
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🎯 Precision-Recall Curves</div>
    <p class="section-subtitle">Trade-off between precision and recall at different classification thresholds</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)

fig_pr = go.Figure()

for idx, row in results_df.iterrows():
    precision, recall, _ = precision_recall_curve(y_test, row['y_pred_proba'])
    avg_precision = average_precision_score(y_test, row['y_pred_proba'])
    
    fig_pr.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f"{row['Model']} (AP = {avg_precision:.3f})",
        line=dict(width=3)
    ))

# Baseline
baseline = y_test.sum() / len(y_test)
fig_pr.add_trace(go.Scatter(
    x=[0, 1],
    y=[baseline, baseline],
    mode='lines',
    name=f'Baseline (prevalence = {baseline:.3f})',
    line=dict(dash='dash', color='gray', width=2)
))

fig_pr.update_layout(
    title=dict(
        text="<b>Precision-Recall Curves - All Models</b>",
        font=dict(size=18, color="#1e293b")
    ),
    xaxis_title="Recall (Sensitivity)",
    yaxis_title="Precision",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=12),
    margin=dict(t=60, b=40, l=40, r=40),
    height=500,
    showlegend=True,
    legend=dict(x=0.6, y=0.9)
)

st.plotly_chart(fig_pr, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>💡 Precision-Recall Interpretation:</strong> Precision-Recall curves are especially useful for 
    imbalanced datasets. Precision measures the proportion of positive predictions that are correct, while 
    recall measures the proportion of actual positives that are correctly identified. High values for both 
    indicate excellent model performance.
</div>
""", unsafe_allow_html=True)

# =========================
# Confusion Matrices
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🔢 Confusion Matrices</div>
    <p class="section-subtitle">Detailed breakdown of true/false positives and negatives for each model</p>
</div>
""", unsafe_allow_html=True)

# Create subplots for confusion matrices
n_models = len(results_df)
cols_per_row = 2
n_rows = (n_models + cols_per_row - 1) // cols_per_row

fig_cm = make_subplots(
    rows=n_rows,
    cols=cols_per_row,
    subplot_titles=[row['Model'] for _, row in results_df.iterrows()],
    vertical_spacing=0.15,
    horizontal_spacing=0.1
)

for idx, row in results_df.iterrows():
    cm = confusion_matrix(y_test, row['y_pred'])
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Determine position
    row_pos = idx // cols_per_row + 1
    col_pos = idx % cols_per_row + 1
    
    # Add heatmap
    fig_cm.add_trace(
        go.Heatmap(
            z=cm_normalized,
            x=['Predicted Healthy', 'Predicted Cancer'],
            y=['Actual Healthy', 'Actual Cancer'],
            text=[[f'{cm[i][j]}<br>({cm_normalized[i][j]:.1%})' for j in range(2)] for i in range(2)],
            texttemplate='%{text}',
            textfont={"size": 12},
            colorscale='Blues',
            showscale=(idx == 0),
            colorbar=dict(title="Proportion", x=1.15) if idx == 0 else None
        ),
        row=row_pos,
        col=col_pos
    )

fig_cm.update_layout(
    title=dict(
        text="<b>Confusion Matrices - All Models</b>",
        font=dict(size=18, color="#1e293b")
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=11),
    margin=dict(t=80, b=40, l=40, r=40),
    height=350 * n_rows,
    showlegend=False
)

st.plotly_chart(fig_cm, use_container_width=True)

# =========================
# Calibration Curves
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">⚖️ Calibration Curves</div>
    <p class="section-subtitle">Assessing how well predicted probabilities match actual outcomes</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)

fig_calib = go.Figure()

for idx, row in results_df.iterrows():
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, row['y_pred_proba'], n_bins=10, strategy='uniform'
    )
    
    fig_calib.add_trace(go.Scatter(
        x=mean_predicted_value,
        y=fraction_of_positives,
        mode='lines+markers',
        name=row['Model'],
        line=dict(width=3),
        marker=dict(size=8)
    ))

# Perfect calibration line
fig_calib.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Perfect Calibration',
    line=dict(dash='dash', color='gray', width=2)
))

fig_calib.update_layout(
    title=dict(
        text="<b>Calibration Curves - Predicted vs Actual Probabilities</b>",
        font=dict(size=18, color="#1e293b")
    ),
    xaxis_title="Mean Predicted Probability",
    yaxis_title="Fraction of Positives",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=12),
    margin=dict(t=60, b=40, l=40, r=40),
    height=500,
    showlegend=True
)

st.plotly_chart(fig_calib, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>💡 Calibration Interpretation:</strong> A well-calibrated model's predicted probabilities 
    should match the observed frequencies. If a model predicts 70% probability of cancer for a group of 
    patients, approximately 70% of that group should actually have cancer. Curves closer to the diagonal 
    indicate better calibration.
</div>
""", unsafe_allow_html=True)

# =========================
# Feature Importance (for tree-based models)
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🎯 Feature Importance Analysis</div>
    <p class="section-subtitle">Most influential variables for tree-based models</p>
</div>
""", unsafe_allow_html=True)

tree_models = [name for name in selected_models if name in ["Random Forest", "Gradient Boosting"]]

if len(tree_models) > 0:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    fig_fi = go.Figure()
    
    for model_name in tree_models:
        model = models_dict[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = feature_cols
            
            # Sort
            indices = np.argsort(importances)[::-1][:15]  # Top 15
            
            fig_fi.add_trace(go.Bar(
                x=importances[indices],
                y=[feature_names[i] for i in indices],
                orientation='h',
                name=model_name
            ))
    
    fig_fi.update_layout(
        title=dict(
            text="<b>Top 15 Most Important Features</b>",
            font=dict(size=18, color="#1e293b")
        ),
        xaxis_title="Feature Importance",
        yaxis_title="",
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#334155", size=12),
        margin=dict(t=60, b=40, l=150, r=40),
        height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="warning-box">
        <strong>ℹ️ Feature Importance Unavailable:</strong> Feature importance analysis requires tree-based 
        models (Random Forest or Gradient Boosting). Select these models to view this analysis.
    </div>
    """, unsafe_allow_html=True)

# =========================
# Cross-Validation Results
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🔄 Cross-Validation Stability</div>
    <p class="section-subtitle">Performance consistency across multiple data splits</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)

fig_cv = go.Figure()

# Perform detailed CV for visualization
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

for model_name, model in models_dict.items():
    if model_name in ["Support Vector Machine", "Logistic Regression"]:
        X_cv = X_train_scaled
    else:
        X_cv = X_train.values
    
    cv_scores = []
    for train_idx, val_idx in cv.split(X_cv, y_train):
        X_tr, X_val = X_cv[train_idx], X_cv[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred_proba)
        cv_scores.append(score)
    
    fig_cv.add_trace(go.Box(
        y=cv_scores,
        name=model_name,
        boxmean='sd'
    ))

fig_cv.update_layout(
    title=dict(
        text=f"<b>Cross-Validation AUC Scores ({cv_folds}-Fold)</b>",
        font=dict(size=18, color="#1e293b")
    ),
    yaxis_title="ROC-AUC Score",
    yaxis=dict(range=[0, 1]),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#334155", size=12),
    margin=dict(t=60, b=40, l=40, r=40),
    height=500,
    showlegend=False
)

st.plotly_chart(fig_cv, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>💡 Cross-Validation Purpose:</strong> K-fold cross-validation assesses model stability by training 
    and testing on different data subsets. Lower variance (tighter boxes) indicates more stable performance. 
    This helps detect overfitting and ensures generalization to unseen data.
</div>
""", unsafe_allow_html=True)

# =========================
# Model Recommendations
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🎯 Model Selection Recommendations</div>
    <p class="section-subtitle">Data-driven guidance for choosing the optimal algorithm</p>
</div>
""", unsafe_allow_html=True)

col_rec1, col_rec2 = st.columns(2)

with col_rec1:
    st.markdown(f"""
    <div class="chart-container">
        <div style="text-align: center; margin-bottom: 16px;">
            <div style="font-size: 2.5rem;">🏆</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #1e293b; margin-bottom: 12px;">
                Best Overall Performance
            </div>
        </div>
        <div style="color: #475569; line-height: 1.7; font-size: 0.95rem;">
            Based on ROC-AUC scores, <strong>{best_model_name}</strong> is the top performer with 
            AUC = <strong>{best_auc:.3f}</strong>. This model demonstrates the best balance between 
            sensitivity and specificity, making it the recommended choice for lung cancer risk prediction 
            in this dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_rec2:
    # Find most stable model (lowest CV std)
    most_stable_idx = results_df['CV Std'].idxmin()
    most_stable_name = results_df.loc[most_stable_idx, 'Model']
    stable_cv_std = results_df.loc[most_stable_idx, 'CV Std']
    
    st.markdown(f"""
    <div class="chart-container">
        <div style="text-align: center; margin-bottom: 16px;">
            <div style="font-size: 2.5rem;">📊</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #1e293b; margin-bottom: 12px;">
                Most Stable Performance
            </div>
        </div>
        <div style="color: #475569; line-height: 1.7; font-size: 0.95rem;">
            <strong>{most_stable_name}</strong> shows the lowest cross-validation variance 
            (Std = <strong>{stable_cv_std:.4f}</strong>), indicating consistent performance across 
            different data splits. This suggests reliable generalization to new patient data.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Clinical context recommendations
st.markdown("""
<div class="warning-box">
    <strong>⚕️ Clinical Application Considerations:</strong><br><br>
    
    <strong>For Screening Programs:</strong><br>
    • Prioritize <strong>high sensitivity (recall)</strong> to minimize false negatives<br>
    • Accept lower specificity to avoid missing potential cancer cases<br>
    • Consider threshold adjustment to increase recall at the cost of precision<br><br>
    
    <strong>For Diagnostic Support:</strong><br>
    • Balance sensitivity and specificity based on clinical context<br>
    • Use probability scores for risk stratification rather than binary classification<br>
    • Combine model predictions with clinical expertise and additional diagnostic tests<br><br>
    
    <strong>For Resource Allocation:</strong><br>
    • Prioritize <strong>high precision</strong> to efficiently allocate limited screening resources<br>
    • Focus on patients with highest predicted risk for intensive follow-up<br>
    • Use calibrated probabilities to guide intervention intensity
</div>
""", unsafe_allow_html=True)

# =========================
# Threshold Analysis
# =========================
st.markdown("""
<div style="margin: 50px 0 10px 0;">
    <div class="section-header">🎚️ Classification Threshold Analysis</div>
    <p class="section-subtitle">Interactive threshold tuning for optimal clinical decision-making</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.markdown('<div class="control-title">⚙️ Threshold Configuration</div>', unsafe_allow_html=True)

col_thresh1, col_thresh2 = st.columns([2, 1])

with col_thresh1:
    selected_model_threshold = st.selectbox(
        "Select model for threshold analysis",
        results_df['Model'].tolist(),
        index=best_model_idx,
        help="Choose which model to analyze"
    )

with col_thresh2:
    threshold = st.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Probability threshold for positive classification"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Get selected model's predictions
selected_row = results_df[results_df['Model'] == selected_model_threshold].iloc[0]
y_pred_proba_selected = selected_row['y_pred_proba']

# Apply threshold
y_pred_thresh = (y_pred_proba_selected >= threshold).astype(int)

# Calculate metrics at this threshold
accuracy_thresh = accuracy_score(y_test, y_pred_thresh)
precision_thresh = precision_score(y_test, y_pred_thresh, zero_division=0)
recall_thresh = recall_score(y_test, y_pred_thresh)
f1_thresh = f1_score(y_test, y_pred_thresh, zero_division=0)

# Confusion matrix
cm_thresh = confusion_matrix(y_test, y_pred_thresh)
tn, fp, fn, tp = cm_thresh.ravel()

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Accuracy</div>
        <div class="stat-value">{accuracy_thresh:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Precision (PPV)</div>
        <div class="stat-value">{precision_thresh:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Recall (Sensitivity)</div>
        <div class="stat-value">{recall_thresh:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">F1-Score</div>
        <div class="stat-value">{f1_thresh:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

# Confusion matrix at threshold
st.markdown('<div class="chart-container">', unsafe_allow_html=True)

col_cm1, col_cm2 = st.columns([1, 1])

with col_cm1:
    cm_normalized = cm_thresh.astype('float') / cm_thresh.sum(axis=1)[:, np.newaxis]
    
    fig_cm_thresh = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=['Predicted Healthy', 'Predicted Cancer'],
        y=['Actual Healthy', 'Actual Cancer'],
        text=[[f'{cm_thresh[i][j]}<br>({cm_normalized[i][j]:.1%})' for j in range(2)] for i in range(2)],
        texttemplate='%{text}',
        textfont={"size": 14},
        colorscale='Blues',
        showscale=True
    ))
    
    fig_cm_thresh.update_layout(
        title=dict(
            text=f"<b>Confusion Matrix (Threshold = {threshold:.2f})</b>",
            font=dict(size=16, color="#1e293b")
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#334155", size=12),
        margin=dict(t=60, b=40, l=40, r=40),
        height=400
    )
    
    st.plotly_chart(fig_cm_thresh, use_container_width=True)

with col_cm2:
    # Clinical metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    st.markdown(f"""
    <div style="padding: 20px;">
        <div style="font-weight: 700; font-size: 1.1rem; color: #1e293b; margin-bottom: 20px;">
            Clinical Metrics Breakdown
        </div>
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 4px;">True Positives (TP)</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{tp}</div>
            <div style="font-size: 0.8rem; color: #64748b;">Cancer correctly identified</div>
        </div>
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 4px;">True Negatives (TN)</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{tn}</div>
            <div style="font-size: 0.8rem; color: #64748b;">Healthy correctly identified</div>
        </div>
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 4px;">False Positives (FP)</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">{fp}</div>
            <div style="font-size: 0.8rem; color: #64748b;">Healthy misclassified as cancer</div>
        </div>
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 4px;">False Negatives (FN)</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">{fn}</div>
            <div style="font-size: 0.8rem; color: #64748b;">Cancer missed</div>
        </div>
        <hr style="border: 1px solid #e2e8f0; margin: 20px 0;">
        <div style="margin-bottom: 12px;">
            <span style="color: #64748b;">Specificity:</span>
            <strong style="color: #2563eb; float: right;">{specificity:.3f}</strong>
        </div>
        <div>
            <span style="color: #64748b;">NPV (Neg. Pred. Value):</span>
            <strong style="color: #2563eb; float: right;">{npv:.3f}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="info-box">
    <strong>🎯 Threshold Impact:</strong> At threshold = {threshold:.2f}, the model classifies 
    <strong>{(y_pred_proba_selected >= threshold).sum()}</strong> patients as high-risk. 
    {'Lowering the threshold increases sensitivity (catches more cancer cases) but decreases precision (more false alarms). ' if threshold > 0.5 else 'Raising the threshold increases precision (fewer false alarms) but decreases sensitivity (may miss some cancer cases). '}
    Adjust based on clinical priorities and resource constraints.
</div>
""", unsafe_allow_html=True)

# =========================
# Methodology Documentation
# =========================
st.markdown("""
<div style="margin: 60px 0 10px 0;">
    <div class="section-header">📚 Evaluation Methodology</div>
    <p class="section-subtitle">Technical documentation of evaluation procedures and metrics</p>
</div>
""", unsafe_allow_html=True)

col_meth1, col_meth2 = st.columns(2)

with col_meth1:
    st.markdown("""
    <div class="chart-container">
        <div style="font-weight: 700; font-size: 1rem; color: #1e293b; margin-bottom: 16px;">
            🎯 Performance Metrics Definitions
        </div>
        <div style="color: #475569; line-height: 1.8; font-size: 0.9rem;">
            <strong>• Accuracy:</strong> (TP + TN) / Total - Overall correctness<br>
            <strong>• Precision (PPV):</strong> TP / (TP + FP) - Positive prediction accuracy<br>
            <strong>• Recall (Sensitivity):</strong> TP / (TP + FN) - True positive rate<br>
            <strong>• Specificity:</strong> TN / (TN + FP) - True negative rate<br>
            <strong>• F1-Score:</strong> 2 × (Precision × Recall) / (Precision + Recall)<br>
            <strong>• ROC-AUC:</strong> Area under ROC curve (0.5-1.0)<br>
            <strong>• NPV:</strong> TN / (TN + FN) - Negative prediction accuracy
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_meth2:
    st.markdown("""
    <div class="chart-container">
        <div style="font-weight: 700; font-size: 1rem; color: #1e293b; margin-bottom: 16px;">
            🔬 Validation Strategy
        </div>
        <div style="color: #475569; line-height: 1.8; font-size: 0.9rem;">
            <strong>• Train-Test Split:</strong> 75% training, 25% testing<br>
            <strong>• Stratification:</strong> Balanced class distribution in splits<br>
            <strong>• Cross-Validation:</strong> {cv_folds}-fold stratified CV<br>
            <strong>• Randomization:</strong> Fixed seed (42) for reproducibility<br>
            <strong>• Scaling:</strong> StandardScaler for SVM and Logistic Regression<br>
            <strong>• Class Balance:</strong> Original distribution preserved<br>
            <strong>• Hyperparameters:</strong> Default scikit-learn configurations
        </div>
    </div>
    """, unsafe_allow_html=True)

# Final recommendations
st.markdown("""
<div class="warning-box">
    <strong>⚕️ Clinical Implementation Guidelines:</strong><br><br>
    
    <strong>1. Model Deployment:</strong> Use the {0} as the primary prediction engine based on its superior 
    ROC-AUC performance ({1:.3f}). Implement regular retraining on updated patient data.<br><br>
    
    <strong>2. Threshold Selection:</strong> For screening applications, consider threshold ≈ 0.3-0.4 to maximize 
    sensitivity. For diagnostic confirmation, use threshold ≈ 0.6-0.7 to maximize precision.<br><br>
    
    <strong>3. Model Monitoring:</strong> Track performance metrics monthly. If ROC-AUC drops below 0.75 or 
    calibration deteriorates, retrain with recent data.<br><br>
    
    <strong>4. Clinical Integration:</strong> Present predictions as probability scores with confidence intervals. 
    Always combine with clinical judgment, patient history, and additional diagnostic procedures.<br><br>
    
    <strong>5. Ethical Considerations:</strong> Ensure model fairness across demographic groups. Regular audits 
    for bias. Transparent communication of limitations to patients and clinicians.
</div>
""".format(best_model_name, best_auc), unsafe_allow_html=True)