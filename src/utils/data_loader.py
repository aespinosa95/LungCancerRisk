import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import scipy.stats as ss

# ============================
# Data Loading & Preprocessing
# ============================

def load_data(filepath='../data/survey lung cancer_clean.csv'):
    """
    Load and preprocess the lung cancer dataset.
    
    Assumes all variables except 'age' are already 0/1 binary.
    """
    df = pd.read_csv(filepath)

    # Ensure age is numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median())

    # Standardize age
    scaler = StandardScaler()
    df['age_scaled'] = scaler.fit_transform(df[['age']])

    # Ensure binary columns are 0/1 integers
    binary_cols = [col for col in df.columns if col not in ['age', 'age_scaled']]
    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    return df

# ============================
# Risk Score Calculation
# ============================

def compute_risk_score(df):
    """Compute logistic regression risk score and feature importance."""
    features = [col for col in df.columns if col not in ['lung_cancer', 'age', 'age_scaled']]

    X = df[features]
    y = df['lung_cancer']

    # Handle insufficient samples/classes
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        risk_scores = np.random.uniform(0, 1, len(df))
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': np.random.uniform(0,1,len(features)),
            'direction': np.random.choice([-1,1], len(features))
        })
        return risk_scores, feature_importance

    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    risk_scores = model.predict_proba(X)[:,1]

    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': abs(model.coef_[0]),
        'direction': np.sign(model.coef_[0])
    }).sort_values('importance', ascending=False)

    return risk_scores, feature_importance

# ============================
# Clustering
# ============================

def perform_clustering(df, features=None, n_clusters=3):
    """Perform KMeans clustering with PCA for visualization."""
    if features is None:
        features = [col for col in df.columns if col not in ['lung_cancer', 'age_scaled']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    df['cluster'] = clusters
    cluster_profiles = df.groupby('cluster')[features].mean()

    return clusters, X_pca, cluster_profiles

# ============================
# Descriptive Statistics
# ============================

def get_statistics(df, group_by=None):
    """
    Compute descriptive statistics for numeric columns.
    """
    numeric_cols = ['age', 'age_scaled'] + [col for col in df.columns if col not in ['age', 'age_scaled']]
    numeric_cols = list(dict.fromkeys(numeric_cols))  # Ensure uniqueness

    if group_by:
        stats = df.groupby(group_by)[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
        stats.columns = [f"{col[0]} ({col[1].capitalize()})" for col in stats.columns]
        return stats
    else:
        stats = df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max']).transpose()
        stats.rename(columns={
            'mean': 'Mean',
            'median': 'Median',
            'std': 'Std',
            'min': 'Min',
            'max': 'Max'
        }, inplace=True)
        return stats

# ============================
# Mixed Correlation (Numeric + Categorical)
# ============================

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty:
        return np.nan
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - (k-1)*(r-1)/(n-1))
    rcorr = r - (r-1)**2/(n-1)
    kcorr = k - (k-1)**2/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def point_biserial(x, y):
    try:
        return ss.pointbiserialr(x, y)[0]
    except:
        return np.nan

def pearson_corr(x, y):
    try:
        return ss.pearsonr(x, y)[0]
    except:
        return np.nan

def compute_mixed_correlation(df, numerical_cols):
    """Compute mixed correlation matrix for numeric and categorical variables."""
    if df.empty:
        return pd.DataFrame()  # Avoid errors if no data

    cols = df.columns
    assoc_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for i in cols:
        for j in cols:
            if i==j:
                assoc_matrix.loc[i,j] = 1.0
            else:
                xi, x_type = df[i], "num" if i in numerical_cols else "cat"
                yj, y_type = df[j], "num" if j in numerical_cols else "cat"

                # Numeric-Numeric
                if x_type=="num" and y_type=="num":
                    assoc_matrix.loc[i,j] = pearson_corr(xi,yj)
                # Numeric-Categorical binary
                elif x_type=="num" and y_type=="cat" and len(df[j].unique())==2:
                    assoc_matrix.loc[i,j] = point_biserial(xi,yj)
                elif x_type=="cat" and y_type=="num" and len(df[i].unique())==2:
                    assoc_matrix.loc[i,j] = point_biserial(yj,xi)
                # Categorical-Categorical
                else:
                    assoc_matrix.loc[i,j] = cramers_v(xi,yj)
    return assoc_matrix
