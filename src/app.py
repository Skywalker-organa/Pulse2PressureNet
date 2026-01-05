# app.py - Streamlit Web Application
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

from advanced_model import CNN_LSTM_Model
from simple_loader import generate_ppg_from_bp


# ------------------ Page Config ------------------
st.set_page_config(
    page_title="PPG Blood Pressure Estimator",
    page_icon="🫀",
    layout="wide"
)


# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    model = CNN_LSTM_Model()
    model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
    model.eval()
    return model


model = load_model()


# ------------------ Title ------------------
st.title("🫀 PPG-Based Blood Pressure Estimation")
st.markdown("### Cuffless BP monitoring using deep learning")


# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("About")
    st.write("""
    This application estimates blood pressure from photoplethysmography (PPG) signals 
    using a CNN-LSTM deep learning model.
    
    **Performance:**
    - Systolic MAE: ~5.2 mmHg
    - Diastolic MAE: ~4.8 mmHg
    """)

    st.header("Patient Info")
    age = st.number_input("Age", 18, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    heart_rate = st.slider("Heart Rate (bpm)", 40, 150, 70)


# ------------------ Tabs ------------------
tab1, tab2, tab3 = st.tabs(["📊 Demo", "📈 Batch Analysis", "📚 About"])


# =========================================================
# TAB 1 — DEMO
# =========================================================
with tab1:
    st.header("Demo: Generate & Predict")

    col1, col2 = st.columns(2)

    # -------- Left: Generate Signal --------
    with col1:
        st.subheader("Generate Test Signal")

        true_sys = st.slider("True Systolic BP", 80, 200, 120)
        true_dia = st.slider("True Diastolic BP", 40, 120, 80)

        if st.button("Generate PPG Signal", type="primary"):
            ppg = generate_ppg_from_bp(true_sys, true_dia, heart_rate)

            st.session_state["ppg"] = ppg
            st.session_state["true_bp"] = (true_sys, true_dia)

            fig, ax = plt.subplots(figsize=(10, 4))
            time = np.linspace(0, 10, 1250)
            ax.plot(time, ppg, color="darkblue", linewidth=1)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"Generated PPG Signal ({true_sys}/{true_dia} mmHg)")
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

    # -------- Right: Predict BP --------
    with col2:
        st.subheader("Predict Blood Pressure")

        if st.button("Estimate BP", type="primary") and "ppg" in st.session_state:
            ppg = st.session_state["ppg"]
            true_sys, true_dia = st.session_state["true_bp"]

            with st.spinner("Analyzing PPG signal..."):
                ppg_tensor = torch.FloatTensor(ppg).unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    pred = model(ppg_tensor)

                pred_sys = pred[0, 0].item()
                pred_dia = pred[0, 1].item()

                error_sys = abs(pred_sys - true_sys)
                error_dia = abs(pred_dia - true_dia)

            st.success("Prediction Complete!")

            colA, colB = st.columns(2)

            with colA:
                st.metric("Systolic BP", f"{pred_sys:.1f} mmHg", f"{error_sys:.1f} mmHg error")
            with colB:
                st.metric("Diastolic BP", f"{pred_dia:.1f} mmHg", f"{error_dia:.1f} mmHg error")

            # Comparison Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            x = ["Systolic", "Diastolic"]
            true_vals = [true_sys, true_dia]
            pred_vals = [pred_sys, pred_dia]

            x_pos = np.arange(len(x))
            width = 0.35

            ax.bar(x_pos - width / 2, true_vals, width, label="True", color="steelblue")
            ax.bar(x_pos + width / 2, pred_vals, width, label="Predicted", color="coral")

            ax.set_ylabel("Blood Pressure (mmHg)")
            ax.set_title("True vs Predicted BP")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x)
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)


#batch analysis
with tab2:
    st.header("Batch Analysis")

    if not os.path.exists("data/ppg_dataset.pkl"):
        st.error("Dataset not found. Train model first.")
    else:
        with open("data/ppg_dataset.pkl", "rb") as f:
            data = pickle.load(f)

        st.write(f"**Dataset:** {len(data)} samples")

        if st.button("Run Batch Prediction"):
            progress = st.progress(0)

            predictions = []
            true_values = []

            for i, sample in enumerate(data[:20]):
                ppg_tensor = torch.FloatTensor(sample["ppg"]).unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    pred = model(ppg_tensor)

                predictions.append([pred[0, 0].item(), pred[0, 1].item()])
                true_values.append([sample["systolic"], sample["diastolic"]])

                progress.progress((i + 1) / 20)

            predictions = np.array(predictions)
            true_values = np.array(true_values)

            errors = np.abs(predictions - true_values)

            mae_sys = np.mean(errors[:, 0])
            mae_dia = np.mean(errors[:, 1])

            col1, col2 = st.columns(2)
            col1.metric("Systolic MAE", f"{mae_sys:.2f} mmHg")
            col2.metric("Diastolic MAE", f"{mae_dia:.2f} mmHg")

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].scatter(true_values[:, 0], predictions[:, 0], alpha=0.6)
            axes[0].plot([80, 200], [80, 200], "r--")
            axes[0].set_title("Systolic Prediction")
            axes[0].grid(True)

            axes[1].scatter(true_values[:, 1], predictions[:, 1], alpha=0.6, color="coral")
            axes[1].plot([40, 120], [40, 120], "r--")
            axes[1].set_title("Diastolic Prediction")
            axes[1].grid(True)

            plt.tight_layout()
            st.pyplot(fig)


#about section
with tab3:
    st.header("About This Project")

    st.markdown("""
### 🎯 Project Overview
Cuffless blood pressure estimation using PPG and deep learning.

### 🧠 Model
CNN + LSTM architecture

### ⚠️ Disclaimer
Educational research prototype. Not medical advice.
""")

st.markdown("---")
st.caption("⚠️ Research Only. Not FDA Approved.")



