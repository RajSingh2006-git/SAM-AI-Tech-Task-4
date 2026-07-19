import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import time

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection Portal",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism + Modern Dark Accent UI)
st.markdown("""
<style>
    /* Main Theme Overrides */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Card Container */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.2);
    }
    
    /* Headers & Text */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Custom Alert Boxes */
    .fraud-alert {
        background: rgba(239, 68, 68, 0.15);
        border-left: 6px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.2);
    }
    .legit-alert {
        background: rgba(34, 197, 94, 0.15);
        border-left: 6px solid #22c55e;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 4px 20px rgba(34, 197, 94, 0.2);
    }

    /* Metric Values */
    .metric-title {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 5px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #090d16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# Function to generate synthetic credit card dataset if real dataset is not found
@st.cache_data
def generate_synthetic_data(n_legit=4000, n_fraud=200):
    np.random.seed(42)
    v_cols = [f'V{i}' for i in range(1, 29)]
    
    legit_time = np.random.uniform(0, 172800, n_legit)
    legit_amount = np.random.exponential(scale=88, size=n_legit)
    legit_v = np.random.normal(loc=0, scale=1.0, size=(n_legit, 28))
    
    fraud_time = np.random.uniform(0, 172800, n_fraud)
    fraud_amount = np.random.exponential(scale=122, size=n_fraud)
    fraud_v = np.random.normal(loc=0, scale=1.0, size=(n_fraud, 28))
    
    fraud_v[:, 0] -= 3.5
    fraud_v[:, 1] += 2.8
    fraud_v[:, 2] -= 4.1
    fraud_v[:, 3] += 3.2
    fraud_v[:, 11] -= 2.5
    fraud_v[:, 13] -= 3.0
    fraud_v[:, 16] -= 2.8
    
    df_legit = pd.DataFrame(legit_v, columns=v_cols)
    df_legit['Time'] = legit_time
    df_legit['Amount'] = legit_amount
    df_legit['Class'] = 0

    df_fraud = pd.DataFrame(fraud_v, columns=v_cols)
    df_fraud['Time'] = fraud_time
    df_fraud['Amount'] = fraud_amount
    df_fraud['Class'] = 1

    df_combined = pd.concat([df_legit, df_fraud], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    cols = ['Time'] + v_cols + ['Amount', 'Class']
    return df_combined[cols]


# Function to load dataset (CSV upload or local file or synthetic)
@st.cache_data
def load_dataset(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df, "Uploaded CSV"
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
    
    try:
        df = pd.read_csv('creditcard.csv')
        return df, "creditcard.csv (Local)"
    except Exception:
        pass

    df = generate_synthetic_data()
    return df, "Generated Sample Credit Card Dataset"


# Main Header
st.title("💳 Credit Card Fraud Detection System")
st.markdown("<div class='sub-header'>Machine Learning powered real-time fraud detection, dataset analytics, and under-sampling workflow</div>", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/isometric-reflection/100/security-shield.png", width=70)
st.sidebar.title("Navigation & Controls")

uploaded_csv = st.sidebar.file_uploader("📁 Upload Custom Dataset (CSV)", type=['csv'])
df, data_source_name = load_dataset(uploaded_csv)

st.sidebar.info(f"**Current Data Source:** {data_source_name}\n\n**Total Records:** {len(df):,}")

page = st.sidebar.radio(
    "Go to Section",
    [
        "📊 Dataset Overview & EDA",
        "⚖️ Under-Sampling Workflow",
        "🤖 Model Training & Evaluation",
        "⚡ Single Transaction Predictor",
        "📁 Batch File Fraud Detection"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("SAM AI • Credit Card Fraud ML Dashboard")


# ---------------------------------------------------------
# SECTION 1: DATASET OVERVIEW & EDA
# ---------------------------------------------------------
if page == "📊 Dataset Overview & EDA":
    st.header("📊 Exploratory Data Analysis & Statistics")
    
    total_tx = len(df)
    legit_tx = len(df[df['Class'] == 0])
    fraud_tx = len(df[df['Class'] == 1])
    fraud_pct = (fraud_tx / total_tx) * 100
    total_amount = df['Amount'].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Transactions</div>
            <div class="metric-val">{total_tx:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Legit Transactions</div>
            <div class="metric-val" style="color: #22c55e;">{legit_tx:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Fraud Transactions</div>
            <div class="metric-val" style="color: #ef4444;">{fraud_tx:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Fraud Ratio</div>
            <div class="metric-val" style="color: #f59e0b;">{fraud_pct:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Value</div>
            <div class="metric-val" style="color: #3b82f6;">${total_amount:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("⚖️ Target Class Distribution")
        fig_pie = px.pie(
            names=["Legit (0)", "Fraud (1)"],
            values=[legit_tx, fraud_tx],
            hole=0.4,
            color_discrete_sequence=['#22c55e', '#ef4444'],
            title="Class Proportion (Highly Imbalanced)"
        )
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("💰 Transaction Amount Distribution")
        fig_box = px.box(
            df,
            x='Class',
            y='Amount',
            color='Class',
            color_discrete_map={0: '#22c55e', 1: '#ef4444'},
            labels={'Class': 'Transaction Type (0=Legit, 1=Fraud)', 'Amount': 'Amount ($)'},
            title="Transaction Amount by Class"
        )
        fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'))
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 First 5 Rows (Head)", "📑 Last 5 Rows (Tail)", "ℹ️ Missing Values & Info", "📈 Statistical Summary"])
    
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    with tab2:
        st.dataframe(df.tail(10), use_container_width=True)
    with tab3:
        null_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Missing Count': df.isnull().sum().values
        })
        st.dataframe(null_df, use_container_width=True)
    with tab4:
        st.write("**Legit Transactions Amount Summary:**")
        st.dataframe(df[df['Class'] == 0]['Amount'].describe().to_frame().T, use_container_width=True)
        st.write("**Fraudulent Transactions Amount Summary:**")
        st.dataframe(df[df['Class'] == 1]['Amount'].describe().to_frame().T, use_container_width=True)
        st.write("**Mean Feature Comparison by Class (`groupby('Class').mean()`):**")
        st.dataframe(df.groupby('Class').mean(), use_container_width=True)


# ---------------------------------------------------------
# SECTION 2: UNDER-SAMPLING WORKFLOW
# ---------------------------------------------------------
elif page == "⚖️ Under-Sampling Workflow":
    st.header("⚖️ Under-Sampling & Data Balancing")
    st.markdown("""
    Because credit card transaction datasets are heavily **imbalanced** (e.g. 99.8% normal vs 0.2% fraud), 
    a standard machine learning model trained on raw data may become biased towards predicting all transactions as normal.
    
    **Under-Sampling Strategy (as in notebook):**
    We sample a matching number of normal transactions to equal the total count of fraudulent transactions, creating a 50:50 balanced dataset.
    """)

    legit_df = df[df['Class'] == 0]
    fraud_df = df[df['Class'] == 1]
    
    n_fraud = len(fraud_df)
    n_legit = len(legit_df)

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Original Legit Transactions Count:** {n_legit:,}")
    with col2:
        st.warning(f"**Original Fraud Transactions Count:** {n_fraud:,}")

    st.subheader("Interactive Under-Sampling Controls")
    sample_size = st.slider("Select Number of Legit Samples to extract:", min_value=10, max_value=n_legit, value=min(n_fraud, n_legit), step=10)

    if st.button("⚡ Generate Balanced Dataset"):
        legit_sample = legit_df.sample(n=sample_size, random_state=42)
        balanced_df = pd.concat([legit_sample, fraud_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
        
        st.session_state['balanced_df'] = balanced_df
        st.success(f"Successfully created balanced dataset with {len(balanced_df)} total records ({sample_size} Legit + {n_fraud} Fraud)!")

    if 'balanced_df' not in st.session_state:
        legit_sample = legit_df.sample(n=min(n_fraud, n_legit), random_state=42)
        st.session_state['balanced_df'] = pd.concat([legit_sample, fraud_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    b_df = st.session_state['balanced_df']

    st.markdown("### 📊 Balanced Dataset Visual Comparison")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_bar = px.bar(
            x=["Legit (0)", "Fraud (1)"],
            y=[len(b_df[b_df['Class'] == 0]), len(b_df[b_df['Class'] == 1])],
            color=["Legit (0)", "Fraud (1)"],
            color_discrete_map={"Legit (0)": '#22c55e', "Fraud (1)": '#ef4444'},
            title="Class Balance in New Dataset",
            labels={'x': 'Class', 'y': 'Count'}
        )
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown("**Balanced Dataset Mean Feature Summary (`b_df.groupby('Class').mean()`):**")
        st.dataframe(b_df.groupby('Class').mean().iloc[:, :8], use_container_width=True)

    st.markdown("### 📄 Preview of Balanced Dataset")
    st.dataframe(b_df.head(10), use_container_width=True)


# ---------------------------------------------------------
# SECTION 3: MODEL TRAINING & EVALUATION
# ---------------------------------------------------------
elif page == "🤖 Model Training & Evaluation":
    st.header("🤖 Machine Learning Model Training & Evaluation")
    st.markdown("Train and compare Machine Learning classifiers on the balanced transaction dataset.")

    st.subheader("⚙️ Model Configuration Settings")
    col_c1, col_c2, col_c3 = st.columns(3)

    with col_c1:
        use_balanced = st.checkbox("Train on Balanced Dataset (Under-Sampled)", value=True)
    with col_c2:
        model_choice = st.selectbox("Select ML Classifier Model", ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier"])
    with col_c3:
        test_ratio = st.slider("Test Data Split Ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    data_to_use = st.session_state.get('balanced_df', df) if use_balanced else df

    X = data_to_use.drop(columns='Class', axis=1)
    y = data_to_use['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=42
    )

    if st.button("🚀 Train Model Now"):
        with st.spinner("Training model, please wait..."):
            time.sleep(0.5)

            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_choice == "Random Forest Classifier":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = DecisionTreeClassifier(random_state=42)

            model.fit(X_train, y_train)

            st.session_state['trained_model'] = model
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['model_name'] = model_choice

            st.success(f"Model **{model_choice}** trained successfully!")

    if 'trained_model' not in st.session_state:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        st.session_state['trained_model'] = model
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['model_name'] = "Logistic Regression"

    model = st.session_state['trained_model']
    X_tr = st.session_state['X_train']
    X_te = st.session_state['X_test']
    y_tr = st.session_state['y_train']
    y_te = st.session_state['y_test']

    y_train_pred = model.predict(X_tr)
    y_test_pred = model.predict(X_te)
    y_test_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_test_pred

    train_acc = accuracy_score(y_tr, y_train_pred)
    test_acc = accuracy_score(y_te, y_test_pred)
    test_prec = precision_score(y_te, y_test_pred, zero_division=0)
    test_rec = recall_score(y_te, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_te, y_test_pred, zero_division=0)

    st.markdown(f"### 🎯 Evaluation Performance ({st.session_state['model_name']})")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Train Accuracy", f"{train_acc * 100:.2f}%")
    with m2:
        st.metric("Test Accuracy", f"{test_acc * 100:.2f}%")
    with m3:
        st.metric("Precision", f"{test_prec * 100:.2f}%")
    with m4:
        st.metric("Recall", f"{test_rec * 100:.2f}%")
    with m5:
        st.metric("F1 Score", f"{test_f1 * 100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.subheader("📌 Confusion Matrix")
        cm = confusion_matrix(y_te, y_test_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            x=['Legit (0)', 'Fraud (1)'],
            y=['Legit (0)', 'Fraud (1)'],
            color_continuous_scale='Blues',
            title="Confusion Matrix on Test Set"
        )
        fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        st.subheader("📈 ROC Curve & AUC")
        fpr, tpr, _ = roc_curve(y_te, y_test_prob)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.3f})', line=dict(color='#6366f1', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Baseline', line=dict(color='#94a3b8', dash='dash')))
        fig_roc.update_layout(
            title="Receiver Operating Characteristic (ROC)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff')
        )
        st.plotly_chart(fig_roc, use_container_width=True)


# ---------------------------------------------------------
# SECTION 4: SINGLE TRANSACTION PREDICTOR
# ---------------------------------------------------------
elif page == "⚡ Single Transaction Predictor":
    st.header("⚡ Single Transaction Fraud Simulator")
    st.markdown("Test individual transaction parameters in real-time to check for fraudulent behavior.")

    if 'trained_model' not in st.session_state:
        X = df.drop(columns='Class', axis=1)
        y = df['Class']
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        st.session_state['trained_model'] = model

    model = st.session_state['trained_model']

    st.subheader("⚡ Quick Load Preset Sample Data")
    col_p1, col_p2, col_p3 = st.columns(3)

    fraud_samples = df[df['Class'] == 1]
    legit_samples = df[df['Class'] == 0]

    if col_p1.button("🚨 Load Random Fraud Example"):
        row = fraud_samples.sample(1, random_state=np.random.randint(1, 1000)).iloc[0]
        st.session_state['preset_data'] = row.to_dict()
    elif col_p2.button("✅ Load Random Legit Example"):
        row = legit_samples.sample(1, random_state=np.random.randint(1, 1000)).iloc[0]
        st.session_state['preset_data'] = row.to_dict()
    elif col_p3.button("🔄 Reset Inputs"):
        st.session_state['preset_data'] = {}

    preset = st.session_state.get('preset_data', {})

    st.markdown("---")
    st.subheader("📝 Input Transaction Features")

    input_data = {}
    
    col_t, col_a = st.columns(2)
    with col_t:
        input_data['Time'] = st.number_input("Transaction Time (Seconds)", value=float(preset.get('Time', 86400.0)), step=100.0)
    with col_a:
        input_data['Amount'] = st.number_input("Transaction Amount ($)", value=float(preset.get('Amount', 150.0)), step=10.0)

    st.markdown("#### Anonymized PCA Features (V1 to V28)")
    
    v_cols = [f'V{i}' for i in range(1, 29)]
    v_cols_per_row = 4
    
    for i in range(0, len(v_cols), v_cols_per_row):
        cols = st.columns(v_cols_per_row)
        for j, col_name in enumerate(v_cols[i:i+v_cols_per_row]):
            with cols[j]:
                default_val = float(preset.get(col_name, 0.0))
                input_data[col_name] = st.number_input(col_name, value=default_val, format="%.4f")

    ordered_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    input_df = pd.DataFrame([[input_data[c] for c in ordered_cols]], columns=ordered_cols)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Predict Transaction Status", type="primary", use_container_width=True):
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else [1.0 - prediction, float(prediction)]
        
        fraud_prob = probabilities[1] * 100
        legit_prob = probabilities[0] * 100

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h2 style="color: #ef4444; margin: 0;">🚨 FRAUDULENT TRANSACTION DETECTED!</h2>
                    <p style="font-size: 1.1rem; color: #fecaca; margin-top: 10px;">
                        This transaction exhibits suspicious patterns characteristic of financial fraud.
                    </p>
                    <hr style="border-color: rgba(239,68,68,0.3);">
                    <h3 style="color: #ffffff;">Fraud Risk Probability: <span style="color: #ef4444;">{fraud_prob:.1f}%</span></h3>
                    <p><b>Recommended Action:</b> Block transaction immediately and flag card for verification.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-alert">
                    <h2 style="color: #22c55e; margin: 0;">✅ LEGITIMATE TRANSACTION</h2>
                    <p style="font-size: 1.1rem; color: #dcfce7; margin-top: 10px;">
                        This transaction passes standard safety and fraud verification checks.
                    </p>
                    <hr style="border-color: rgba(34,197,94,0.3);">
                    <h3 style="color: #ffffff;">Legitimate Confidence: <span style="color: #22c55e;">{legit_prob:.1f}%</span></h3>
                    <p><b>Recommended Action:</b> Approve transaction.</p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = fraud_prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fraud Risk Score (%)", 'font': {'color': "#ffffff"}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': "#ffffff"},
                    'bar': {'color': "#ef4444" if fraud_prob > 50 else "#22c55e"},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.3)"},
                        {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.3)"},
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'))
            st.plotly_chart(fig_gauge, use_container_width=True)


# ---------------------------------------------------------
# SECTION 5: BATCH FILE FRAUD DETECTION
# ---------------------------------------------------------
elif page == "📁 Batch File Fraud Detection":
    st.header("📁 Batch Transaction CSV Predictor")
    st.markdown("Upload a CSV file containing multiple transactions to perform instant automated fraud risk screening.")

    if 'trained_model' not in st.session_state:
        X = df.drop(columns='Class', axis=1)
        y = df['Class']
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        st.session_state['trained_model'] = model

    model = st.session_state['trained_model']

    st.markdown("### 📥 Download Sample Batch File")
    sample_batch_df = df.drop(columns=['Class']).head(50)
    csv_bytes = sample_batch_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 Download Sample Transactions CSV (50 Rows)",
        data=csv_bytes,
        file_name="sample_credit_card_batch.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("### 📤 Upload Batch CSV File")
    batch_file = st.file_uploader("Choose a CSV file for prediction", type=["csv"], key="batch_upload")

    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            st.success(f"Loaded {len(batch_df):,} transactions from `{batch_file.name}`.")
            
            required_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            missing_cols = [c for c in required_cols if c not in batch_df.columns]
            
            if missing_cols:
                st.error(f"Missing required feature columns in CSV: {missing_cols}")
            else:
                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner("Processing batch transactions..."):
                        X_batch = batch_df[required_cols]
                        predictions = model.predict(X_batch)
                        
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(X_batch)[:, 1]
                        else:
                            probs = predictions.astype(float)

                        result_df = batch_df.copy()
                        result_df['Fraud_Prediction'] = predictions
                        result_df['Fraud_Probability_%'] = (probs * 100).round(2)
                        result_df['Status'] = result_df['Fraud_Prediction'].apply(lambda x: '🚨 FRAUD' if x == 1 else '✅ LEGIT')

                        n_total = len(result_df)
                        n_fraud = (predictions == 1).sum()
                        n_legit = (predictions == 0).sum()
                        total_fraud_amt = result_df[result_df['Fraud_Prediction'] == 1]['Amount'].sum()

                        st.markdown("### 📊 Batch Prediction Results Summary")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Batch Size", f"{n_total:,}")
                        c2.metric("Flagged Fraudulent", f"{n_fraud:,}", delta=f"{(n_fraud/n_total)*100:.1f}% of total")
                        c3.metric("Passed Legitimate", f"{n_legit:,}")
                        c4.metric("Flagged Fraud Amount", f"${total_fraud_amt:,.2f}")

                        st.markdown("### 📋 Detailed Predictions Table")
                        st.dataframe(result_df, use_container_width=True)

                        res_csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Fraud Predictions CSV",
                            data=res_csv,
                            file_name="credit_card_fraud_predictions.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
