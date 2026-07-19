# 💳 Credit Card Fraud Detection Web Application

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.10+-3F4F75.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **SAM AI Tech Task 4 Solution**: An end-to-end Machine Learning powered Web Application built with **Streamlit** that converts Jupyter Notebook ML pipeline into an interactive, real-time Fraud Detection Portal.

---

## 📌 Project Overview

This repository contains the full source code and Jupyter Notebook for a **Credit Card Fraud Detection System**. Machine learning models trained on highly imbalanced transaction datasets can struggle to detect fraudulent behavior. This app implements an **Under-Sampling workflow** to balance normal and fraudulent transactions, trains classification models (Logistic Regression, Random Forest, Decision Tree), and provides real-time transaction screening and batch file processing.

---

## 🌟 Key Features

1. **📊 Exploratory Data Analysis & Statistics (EDA)**
   - High-level KPI Metric Cards: Total Transactions, Legit Count, Fraud Count, Fraud Ratio (%), Total Amount.
   - Interactive Class Proportion Pie Chart & Transaction Amount Boxplots.
   - Preview dataset rows, inspect missing values, and analyze feature mean comparisons (`groupby('Class').mean()`).

2. **⚖️ Under-Sampling & Class Balancing**
   - Interactive Under-Sampling strategy matching the Jupyter notebook methodology.
   - Easily extract matching samples of normal transactions to match fraudulent count (creating a 50:50 ratio).
   - Dynamic before-and-after visual distribution bar chart.

3. **🤖 Machine Learning Model Training & Evaluation**
   - Classifiers supported: **Logistic Regression** (Notebook model), **Random Forest Classifier**, and **Decision Tree Classifier**.
   - Adjustable test split ratio and random state parameters.
   - Performance Metrics: Accuracy (Train & Test), Precision, Recall, F1 Score.
   - Interactive **Confusion Matrix Heatmap** and **ROC-AUC Curve**.

4. **⚡ Real-Time Single Transaction Simulator**
   - 1-Click presets: **"Load Random Fraud Example"** and **"Load Random Legit Example"**.
   - Adjustable input parameters for `Time`, `V1`..`V28` (PCA features), and `Amount`.
   - Real-time verdict banner (🚨 Fraud Alert vs ✅ Legit Approval) with interactive Fraud Risk Score Gauge Chart.

5. **📁 Batch CSV File Predictor**
   - Upload custom transaction CSV files for automated bulk risk screening.
   - Download a sample 50-row CSV test template directly from the UI.
   - Summary cards of total flag count, total fraud amount, and downloadable predictions CSV file.

6. **🎲 Built-in Synthetic Sample Generator**
   - Automatic fallback dataset generator so the app runs out-of-the-box even without uploading `creditcard.csv`.

---

## 📂 Project Structure

```
SAM-AI-Tech-Task-4/
├── app.py              # Main Streamlit Web Application
├── notebook.ipynb      # Original Jupyter Notebook for Credit Card Fraud Detection
├── requirements.txt    # Required Python dependencies
└── README.md           # Documentation & Repository Overview
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/RajSingh2006-git/SAM-AI-Tech-Task-4.git
cd SAM-AI-Tech-Task-4
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Model & Dataset Details

- **Dataset Features**:
  - `Time`: Seconds elapsed between each transaction and the first transaction.
  - `V1` – `V28`: Principal components obtained with PCA (for confidentiality).
  - `Amount`: Transaction amount in USD.
  - `Class`: Response variable (1 = Fraud, 0 = Otherwise).
- **Primary Algorithm**: Logistic Regression with random under-sampling for class balancing.

---

## 🛠️ Built With

* [Python](https://www.python.org/)
* [Streamlit](https://streamlit.io/)
* [scikit-learn](https://scikit-learn.org/)
* [Pandas](https://pandas.pydata.org/)
* [Plotly](https://plotly.com/python/)

---

## 👤 Author

**Raj Singh**
- GitHub: [@RajSingh2006-git](https://github.com/RajSingh2006-git)
- Repository: [SAM-AI-Tech-Task-4](https://github.com/RajSingh2006-git/SAM-AI-Tech-Task-4)