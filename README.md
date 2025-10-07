# 🫁 Lung Cancer Risk Analysis Platform

<div align="center">

<img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
<img src="https://img.shields.io/badge/Status-Active-success" alt="Project Status">

---

**Interactive Machine Learning Platform for Lung Cancer Risk Assessment**  
Empowering healthcare professionals with AI-driven insights for early detection and prevention.

[🌐 Live Demo](#) · [🐛 Report Bug](../../issues) · [💡 Request Feature](../../pulls)

---

<img src="https://github.com/aespinosa95/lung_cancer/assets/demo_dashboard.gif" alt="Dashboard Demo" width="800"/>

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset Description](#-dataset-description)
- [Technical Challenges](#️-technical-challenges)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Key Findings](#-key-findings)
- [Dashboard Features](#️-dashboard-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Technologies](#️-technologies)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project delivers a full **data science pipeline** for lung cancer risk prediction, integrating **statistical analysis**, **machine learning**, and **interactive dashboards** built with **Streamlit**.

Healthcare professionals can analyze **clinical risk factors**, visualize **data patterns**, and evaluate **model predictions** through a user-friendly web interface.

### Core Objectives

- 🔍 Identify significant behavioral and clinical risk factors  
- 📊 Develop and validate robust machine learning models  
- 🎨 Build an interactive platform for data exploration  
- 📈 Generate actionable insights for early detection  

---

## 📊 Dataset Description

**Source:** Clinical survey (`survey lung cancer.csv`)  
**Samples:** 309
**Features:** 16 (15 predictors + 1 target)  
**Missing Values:** None  
**Class Imbalance:** 87% cancer, 13% healthy  

| Category | Variables | Type | Description |
|-----------|------------|------|-------------|
| **Demographic** | AGE, GENDER | Continuous, Binary | Age (21–87), sex |
| **Behavioral** | SMOKING, ALCOHOL_CONSUMING, PEER_PRESSURE | Binary | Lifestyle & exposure |
| **Symptoms** | YELLOW_FINGERS, ANXIETY, FATIGUE, ALLERGY, WHEEZING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN | Binary | Clinical symptoms |
| **Comorbidities** | CHRONIC_DISEASE | Binary | Pre-existing conditions |
| **Target** | LUNG_CANCER | Binary | Diagnosis (YES/NO) |

**Encoding:** `1 → NO`, `2 → YES`

---

## ⚙️ Technical Challenges

### 1️⃣ Data Preprocessing

Non-standard binary encoding required conversion:

```python
binary_cols = [...]
df[binary_cols] = df[binary_cols].replace({1: 0, 2: 1}).astype('uint8')
```
✅ Result: Uniform encoding and reduced memory footprint.

---

### 2️⃣ Class Imbalance

**Problem:** 87:13 imbalance → biased predictions.  

---

### 3️⃣ Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
df['age_scaled'] = StandardScaler().fit_transform(df[['age']])
```
---

✅ **Result:** Improved convergence for distance-based models (SVM, Logistic Regression).

---

## 🔬 Exploratory Data Analysis

### 🧓 Age Differences
- Cancer: **62.95 ± 7.97**  
- Healthy: **60.74 ± 9.63**  
- *t-test:* p > 0.05 → Age is not a significant predictor  

---

### 🔝 Top 5 Predictive Symptoms

| Symptom | Healthy (%) | Cancer (%) | Odds Ratio | p-value |
|---------|-------------|------------|------------|---------|
| Allergy | 12.8 | 61.9 | 11.03| < 0.001 |
| Alcohol consume | 17.9 | 61.1 | 7.18 | < 0.001 |
| Swallowing difficulty | 12.8 | 51.9 | 7.32| < 0.001 |
| Wheezing | 23.1 | 60.4 | 5.06 | < 0.001 |
| Coughing | 25.6 | 62.6 | 4.85 | < 0.001 |

---

### 🔗 Correlations

- Anxiety ↔ Yellow Fingers: **r = 0.56**
- Anxiety ↔ swallowing_difficulty: **r = 0.48**  
- Gender ↔ Alcohol consume: **r = 0.45**  
	
---

### 🧩 Clustering Analysis

Unsupervised **K-Means (k=3)** revealed distinct patient profiles:

- **Cluster 0:** Smoking men with yellow fingers and fatigue, but no alcohol consumption
- **Cluster 1:** Wpmen with respiratory symptoms and alcohol consumption.
- **Cluster 2:** Men and women with yellow fingers and difficulty swallowing.

---

## 🎯 Key Findings

- 10 binary symptoms significantly associated (*p < 0.05*)   
- **Top Model:** Gradient Boosting  

| Metric | Score | Interpretation |
|---------|-------|----------------|
| ROC-AUC | 0.959 | Excellent discrimination |
| Accuracy | 91% | High correctness |
| Precision | 96.9% | Few false positives |
| Recall | 92.6% | Few missed cases |
| F1-Score | 0.947 | Balanced performance |

---

## 🖥️ Dashboard Features

| Page | Description |
|------|--------------|
| 🏠 Home | Summary KPIs & dataset overview |
| 📊 Distribution | Histograms, boxplots, descriptive stats |
| 🔄 Correlations | Heatmaps and pairwise associations |
| 📈 Statistical Tests | t-tests, Chi-square, Mann-Whitney U |
| 🎯 Clusters | 3D PCA + radar charts |
| 🧬 Prediction | ML-based risk calculator |
| 🎓 Model Performance | ROC curves, calibration, confusion matrix |

---

### 🎨 Interactive visuals include:
- 3D PCA plots  
- Feature importance bars  
- Radar charts  
- Calibration curves  

---

## 🚀 Installation

```bash
# 1. Clone repository
git clone https://github.com/aespinosa95/lung_cancer.git
cd lung_cancer

# 2. Create virtual environment
python -m venv lungcancer
source lungcancer/bin/activate  # On Windows: lungcancer\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit dashboard
streamlit run src/Home.py
```
## 📁 Project Structure

```text
lung_cancer/
├── data/
│   ├── survey lung cancer.csv
│   └── survey lung cancer_clean.csv
├── src/
│   ├── Home.py
│   ├── pages/
│   └── utils/
├── notebooks/
│   └── preprocess.ipynb
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🛠️ Technologies

| Tool | Purpose |
|------|---------|
| Python 3.12+ | Core language |
| Streamlit | Dashboard framework |
| Pandas / NumPy | Data handling |
| Scikit-learn | ML algorithms |
| Plotly / Matplotlib / Seaborn | Visualization |
| SciPy | Statistical testing |
 

---

## Areas for Contribution

- 🧠 ML model improvements
- 📊 New visualizations
- 🧩 Code optimization
- 🧾 Documentation enhancement

---

## 📄 License

Distributed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

- 🐙 **GitHub:** [@aespinosa95](https://github.com/aespinosa95)  
- 💼 **LinkedIn:** [Asunción Espinosa Sánchez](https://www.linkedin.com/in/asuncion-espinosa-sanchez/)  
- 🌐 **Portfolio:** [asuncionespinosa](https://espinosasa.wixsite.com/portfolio)  
- 📁 **Project Link:** [lung_cancer repo](https://github.com/aespinosa95/lung_cancer)  

---

<div align="center">

⚕️ **Disclaimer:**  
This platform is for **research and educational purposes only**.  
Predictions should be interpreted by qualified healthcare professionals.

*Made with using Streamlit and Python.*

</div>

