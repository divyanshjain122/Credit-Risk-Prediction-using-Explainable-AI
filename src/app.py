import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import xgboost as xgb
import joblib
import os
# === Load model and data ===
@st.cache_resource
def load_assets():
    import os
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(base_dir, "models", "SMOTE_XGBoost.pkl")
    X_train_path = os.path.join(base_dir, "models", "X_train_selected.npy")
    feature_names_path = os.path.join(base_dir, "models", "selected_feature_names.npy")

    model_pipeline = joblib.load(model_path)
    X_train_selected = np.load(X_train_path)  # ✅ use np.load here
    selected_feature_names = np.load(feature_names_path, allow_pickle=True).tolist()
    if X_train_selected.ndim == 1:
        X_train_selected = X_train_selected.reshape(1, -1)


    return model_pipeline, X_train_selected, selected_feature_names

model_pipeline, X_train_selected, selected_feature_names = load_assets()


# === Feature Input Form ===
st.title("Credit Default Prediction Explanation")

st.header("Input Features")
custom_input = {}
defaults = {
    'LIMIT_BAL': 100000,
    'AGE': 30,
    'PAY_1': 0,
    'PAY_2': 2,
    'BILL_AMT1': 5000,
    'BILL_AMT2': 6000,
    'BILL_AMT3': 7000,
    'BILL_AMT4': 8000,
    'BILL_AMT5': 4000,
    'BILL_AMT6': 3000,
    'PAY_AMT1': 1000,
    'PAY_AMT2': 2000,
    'PAY_AMT3': 1500,
    'PAY_AMT4': 1000,
    'PAY_AMT5': 1000,
    'PAY_AMT6': 2000
}

for feature in selected_feature_names:
    default_val = defaults.get(feature, 0)
    custom_input[feature] = st.number_input(feature, value=default_val)


if st.button("Generate Explanation"):

    # === Prepare input ===
    input_df = pd.DataFrame([custom_input])[selected_feature_names]
    input_array_unscaled = input_df.values

    scaler = model_pipeline.named_steps['scaler']
    model = model_pipeline.named_steps['clf']
    scaled_input = scaler.transform(input_array_unscaled)

    dmatrix = xgb.DMatrix(scaled_input, feature_names=selected_feature_names)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    feature_contribs = contribs[0][:-1]
    bias = contribs[0][-1]
    log_odds = bias + np.sum(feature_contribs)
    probability = expit(log_odds)

    # === Display Probability
    st.subheader(f"📊 Probability of Default: **{probability:.3f}**")

    # === Plot 1: TreeInterpreter-style bar plot ===
    st.subheader("1. Feature Contributions")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.barh(selected_feature_names, feature_contribs)
    ax1.axvline(0, color='black')
    ax1.set_title("XGBoost Local Contributions")
    ax1.set_xlabel("Contribution to Log-Odds")
    ax1.grid(True)
    st.pyplot(fig1)

    

    # === Plot 2: Waterfall Plot ===
    st.subheader("2. SHAP Waterfall Plot")
    labels = [f"{name} = {custom_input[name]}" for name in selected_feature_names]
    contrib_pairs = list(zip(labels, feature_contribs))
    contrib_pairs.sort(key=lambda x: abs(x[1]))

    base = bias
    positions = [base]
    values = []
    for _, contrib in contrib_pairs:
        next_val = positions[-1] + contrib
        positions.append(next_val)
        values.append(contrib)

    fig3, ax3 = plt.subplots(figsize=(12, 8))
    for i, (label, value) in enumerate(zip(contrib_pairs, values)):
        color = '#2ca02c' if value > 0 else '#d62728'
        left = positions[i]
        ax3.barh(i, value, left=left, color=color, edgecolor='black', height=0.8)
        ax3.text(left + value / 2, i, f"{value:+.3f}", va='center', ha='center', color='white')

    y_labels = [pair[0] for pair in contrib_pairs]
    ax3.set_yticks(np.arange(len(y_labels)))
    ax3.set_yticklabels(y_labels)
    ax3.invert_yaxis()
    ax3.axvline(bias, color='gray', linestyle='--', linewidth=2, label=f"Base (log-odds) = {bias:.3f}")
    ax3.axvline(log_odds, color='blue', linestyle='-', linewidth=2,
                label=f"Log-odds = {log_odds:.3f}\nProb = {probability:.3f}")
    ax3.set_title("Waterfall Plot of Contributions")
    ax3.set_xlabel("Log-Odds Contribution")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)
# === Plot 3: Force Plot-style Explanation ===
