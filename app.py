# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║           CARDIOPREDICT AI  –  Heart Disease Prediction System               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { 
    padding: 10px 14px; border-radius: 8px; 
    transition: background 0.2s; display: block; cursor: pointer;
}
[data-testid="stSidebar"] .stRadio label:hover { background: #334155; }

.metric-card {
    background: linear-gradient(135deg, #6A0DAD 0%, #3B0764 100%);
    border: 1px solid #9333EA;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}

.metric-card:hover { transform: translateY(-2px); }

.metric-value { font-size: 2rem; font-weight: 700; color: #F5F3FF; }
.metric-label { font-size: 0.85rem; color: #E9D5FF; margin-top: 4px; }

/* ── Risk badges ── */
.risk-low    { background:#064e3b; color:#34d399; border:1px solid #34d399; padding:8px 20px; border-radius:20px; font-weight:600; }
.risk-mild   { background:#713f12; color:#fbbf24; border:1px solid #fbbf24; padding:8px 20px; border-radius:20px; font-weight:600; }
.risk-mod    { background:#7c2d12; color:#f97316; border:1px solid #f97316; padding:8px 20px; border-radius:20px; font-weight:600; }
.risk-high   { background:#450a0a; color:#f87171; border:1px solid #f87171; padding:8px 20px; border-radius:20px; font-weight:600; }

.section-title { font-size:1.4rem; font-weight:700; color:#38bdf8; border-bottom:2px solid #334155; padding-bottom:8px; margin-bottom:16px; }

.pred-box { border-radius: 14px; padding: 24px; text-align: center; box-shadow: 0 8px 30px rgba(0,0,0,0.4); }
.pred-box h2 { margin:0; font-size: 1.6rem; }

.rec-card { border-left: 4px solid; border-radius: 8px; padding: 16px 20px; margin: 12px 0; background: #1e293b; }
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load artefacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    feature_names = joblib.load("models/feature_names.pkl")
    X_test, y_test = joblib.load("models/test_data.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    with open("models/roc_data.json") as f:
        roc_data = json.load(f)
    models = {
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "XGBoost": joblib.load("models/xgboost.pkl"),
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "KNN": joblib.load("models/knn.pkl"),
        "SVM": joblib.load("models/svm.pkl"),
    }
    shap_data = {
        "Random Forest": joblib.load("models/random_forest_shap.pkl"),
        "XGBoost": joblib.load("models/xgboost_shap.pkl"),
        "Logistic Regression": joblib.load("models/logistic_regression_shap.pkl"),
        "KNN": joblib.load("models/knn_shap.pkl"),
        "SVM": joblib.load("models/svm_shap.pkl"),
    }
    return feature_names, X_test, y_test, preprocessor, metrics, roc_data, models, shap_data

feature_names, X_test, y_test, preprocessor, metrics, roc_data, models, shap_data = load_all()
MODEL_NAMES = list(models.keys())

# ─── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='text-align:center; padding: 16px 0 24px 0;'>", unsafe_allow_html=True)
    try:
        st.image("logo.png", use_container_width=True)
    except Exception:
        st.markdown("<h2 style='text-align:center; color:#38bdf8;'>🩺 CardioPredict</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='font-size:1.3rem; font-weight:700; color:#38bdf8; margin-top:10px;'>Chuka University</div>
    <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>Heart Disease Prediction System</div>
    <div style='font-size:0.7rem; color:#475569; margin-top:8px;'>Clinical Decision Support</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Overview", "🩺  Prediction", "📊  Model Performance", "📈  ROC Curves", "🧠  XAI — SHAP Analysis", "🗂️  Data Explorer", "📋  About & Methods"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:#e2e8f0; font-weight:700; text-transform:uppercase; text-align:center; margin-top: 15px; line-height: 1.4;'>Center for Data Analytics and Modeling (CDAM)</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("<h1 style='text-align: center; color: #ef4444; font-size: 42px;'>🩺 CardioPredict AI - Heart Disease Clinical Decision Support Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-size: 28px;'><i>Predictive Modelling with Explainable AI for Heart Disease Risk Assessment</i></h3>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    best_auc = max(v["ROC-AUC"] for v in metrics.values())
    best_acc = max(v["Accuracy"] for v in metrics.values())
    best_f1  = max(v["F1 Score"] for v in metrics.values())
    with col1: st.markdown(f'<div class="metric-card"><div class="metric-value">{best_auc:.3f}</div><div class="metric-label">Best ROC-AUC</div></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card"><div class="metric-value">{best_acc:.1%}</div><div class="metric-label">Best Accuracy</div></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-card"><div class="metric-value">{best_f1:.3f}</div><div class="metric-label">Best F1 Score</div></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="metric-card"><div class="metric-value">5</div><div class="metric-label">Trained Models</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### 🎯 Project Objective")
        st.info("This dashboard implements **five machine learning classifiers** to predict heart disease risk from clinical measurements. It combines predictive modelling with **SHAP-based Explainable AI** to provide transparent, clinically meaningful insights.")
        st.markdown("### 🔬 Clinical Features Used")
        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Clinical Role": ["Age-related risk accumulation", "Blood pressure measurement at rest", "Total serum cholesterol level", "Maximum heart rate achieved", "ST depression induced by exercise", "Number of major vessels colored by fluoroscopy", "Biological sex", "Type of9 chest pain experienced", "Fasting blood sugar > 120 mg/dl", "Resting electrocardiographic results", "Exercise induced angina", "Slope of the peak exercise ST segment", "Thalassemia type"]
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("### 🏆 Model Leaderboard")
        rows = [{"Model": name, "AUC": m["ROC-AUC"], "Accuracy": m["Accuracy"], "F1": m["F1 Score"], "Recall": m["Recall"]} for name, m in metrics.items()]
        lb = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
        lb.index += 1
        st.dataframe(lb.style.background_gradient(subset=["AUC","Accuracy","F1","Recall"], cmap="Blues"), use_container_width=True)
        st.markdown("### ⚠️ Risk Stratification\n| Probability | Risk Level | Action |\n|---|---|---|\n| 0–30% | 🟢 Low | Routine care & primary prevention |\n| 31–50% | 🟡 Mild | Preventive action & non-invasive testing |\n| 51–70% | 🟠 Moderate | Urgent cardiology evaluation |\n| 71–100% | 🔴 High | Immediate intervention & invasive diagnostics |")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🩺  Prediction":
    st.markdown("## 🩺 Patient Risk Prediction")
    st.markdown("Enter patient clinical values and select a model to obtain a heart disease risk prediction with SHAP explanation.")
    st.markdown("---")

    sel_model = st.selectbox("🤖 Select Prediction Model", MODEL_NAMES, help="Choose the ML model to use for prediction")
    st.markdown("### 🧾 Patient Clinical Input")

    refs = {"Age": (29, 77, 55), "RestBP": (90, 200, 130), "Chol": (126, 564, 250), "MaxHR": (71, 202, 150), "Oldpeak": (0.0, 6.2, 1.0), "Ca": (0, 3, 0), "Sex": (0, 1, 1), "ChestPain": ["typical", "nontypical", "nonanginal", "asymptomatic"], "Fbs": (0, 1, 0), "RestECG": (0, 2, 0), "ExAng": (0, 1, 0), "Slope": (1, 3, 2), "Thal": ["normal", "fixed", "reversable"]}

    with st.form("patient_form"):
        st.subheader("Patient Demographics & Vitals")
        r1c1, r1c2, r1c3 = st.columns(3)
        r2c1, r2c2, r2c3 = st.columns(3)
        r3c1, r3c2, r3c3 = st.columns(3)

        with r1c1: age = st.number_input("Age (years)", min_value=29, max_value=77, value=refs["Age"][2])
        with r1c2: sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
        with r1c3: chest_pain = st.selectbox("Chest Pain Type", refs["ChestPain"])
        with r2c1: rest_bp = st.number_input("Resting BP (mmHg)", min_value=90, max_value=200, value=refs["RestBP"][2])
        with r2c2: chol = st.number_input("Cholesterol (mg/dL)", min_value=126, max_value=564, value=refs["Chol"][2])
        with r2c3: fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        with r3c1: max_hr = st.number_input("Max Heart Rate (bpm)", min_value=71, max_value=202, value=refs["MaxHR"][2])
        with r3c2: ex_ang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        with r3c3: oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.2, value=refs["Oldpeak"][2], step=0.1)
            
        st.subheader("Diagnostic Test Results")
        r4c1, r4c2, r4c3 = st.columns(3)
        with r4c1: rest_ecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ["Normal (0)", "ST-T abnormality (1)", "LV hypertrophy (2)"][x])
        with r4c2: slope = st.selectbox("ST Segment Slope", [1, 2, 3], format_func=lambda x: ["Upsloping (1)", "Flat (2)", "Downsloping (3)"][x-1])
        with r4c3: 
            ca = st.number_input("Major Vessels (Ca)", min_value=0, max_value=3, value=refs["Ca"][2])
            thal = st.selectbox("Thalassemia (Thal)", refs["Thal"])
            
        submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True, type="primary")

    if submitted:
        patient = pd.DataFrame([{'Age': age, 'Sex': sex, 'ChestPain': chest_pain, 'RestBP': rest_bp, 'Chol': chol, 'Fbs': fbs, 'RestECG': rest_ecg, 'MaxHR': max_hr, 'ExAng': ex_ang, 'Oldpeak': oldpeak, 'Slope': slope, 'Ca': ca, 'Thal': thal}])
        model = models[sel_model]
        prob = model.predict_proba(patient)[0][1]
        pred = int(prob >= 0.5)

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        r1, r2, r3 = st.columns([1, 1.4, 1])
        with r1:
            if pred == 1:
                st.markdown(f'<div class="pred-box" style="background:linear-gradient(135deg,#450a0a,#7f1d1d);"><h2 style="color:#f87171;">⚠️ Heart Disease</h2><p style="color:#fca5a5;">Risk probability: <b>{prob:.1%}</b></p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pred-box" style="background:linear-gradient(135deg,#052e16,#064e3b);"><h2 style="color:#34d399;">✅ No Heart Disease</h2><p style="color:#6ee7b7;">Risk probability: <b>{prob:.1%}</b></p></div>', unsafe_allow_html=True)

        with r2:
            fig_g, ax_g = plt.subplots(figsize=(5, 1.2))
            fig_g.patch.set_facecolor("#0f172a"); ax_g.set_facecolor("#0f172a")
            zones = [(0.30, "#34d399"), (0.20, "#fbbf24"), (0.20, "#f97316"), (0.30, "#f87171")]
            left = 0
            for width, color in zones:
                ax_g.barh(0, width, left=left, color=color, height=0.5, alpha=0.4); left += width
            ax_g.axvline(prob, color="white", linewidth=3)
            ax_g.text(prob, 0.35, f"{prob:.1%}", color="white", ha="center", fontsize=12, fontweight="bold")
            ax_g.set_xlim(0, 1); ax_g.set_ylim(-0.5, 0.7); ax_g.axis("off")
            st.pyplot(fig_g, use_container_width=True); plt.close()

        with r3:
            pct = prob * 100
            if pct <= 30: badge, label = "risk-low", "🟢 LOW RISK"
            elif pct <= 50: badge, label = "risk-mild", "🟡 MILD RISK"
            elif pct <= 70: badge, label = "risk-mod", "🟠 MODERATE RISK"
            else: badge, label = "risk-high", "🔴 HIGH RISK"
            st.markdown(f'<div class="{badge}" style="text-align:center;margin-top:10px;">{label}</div>', unsafe_allow_html=True)
            st.metric("Heart Disease Probability", f"{prob:.1%}")
            st.metric("Model Used", sel_model)

        st.markdown("### 💡 Clinical Recommendations")
        if pct <= 30:
            st.success("**Low Risk (0–30%)**\n- ✅ Continue routine annual screening\n- ✅ Maintain healthy diet and regular physical activity\n- ✅ Monitor weight and cardiovascular risk factors annually")
        elif pct <= 50:
            st.warning("**Mild Risk (31–50%)**\n- ⚠️ Implement structured lifestyle intervention\n- ⚠️ Consider scheduling a non-invasive functional stress test\n- ⚠️ Aggressively manage modifiable risk factors")
        elif pct <= 70:
            st.error("**Moderate Risk (51–70%)**\n- 🔴 Prompt referral to cardiology\n- 🔴 Consider advanced non-invasive imaging (e.g., CCTA)\n- 🔴 Evaluate the need for guideline-directed medical therapy")
        else:
            st.error("**High Risk (71–100%)**\n- 🚨 Immediate cardiology consultation required\n- 🚨 Prepare for potential invasive coronary angiography\n- 🚨 If patient presents with unstable symptoms, admit for inpatient observation")

        # ── SHAP waterfall for this patient ───────────────────────────────────
        st.markdown("### 🧠 SHAP Explanation — Why this prediction?")
        sd = shap_data[sel_model]
        sv = sd["shap_values_positive"]
        ev = sd["expected_value"]
        X_sample = sd["X_test_sample"]
        is_tree_based = sd.get("is_tree_based", False)

        patient_trans = models[sel_model].named_steps['preprocessor'].transform(patient)
        patient_arr = np.array(patient_trans)
        
        if is_tree_based:
            explainer = shap.TreeExplainer(models[sel_model].named_steps['model'])
            p_shap = explainer.shap_values(patient_trans)
            if isinstance(p_shap, list):
                p_shap = p_shap[1] if len(p_shap) > 1 else p_shap[0]
            patient_shap = np.array(p_shap).flatten()[:len(feature_names)].tolist()
        else:
            diffs = np.abs(X_sample.values - patient_arr).sum(axis=1)
            closest_idx = int(np.argmin(diffs))
            patient_shap = sv[closest_idx]
            patient_shap = np.array(patient_shap).flatten()[:len(feature_names)].tolist()

        patient_vals = np.array(patient_arr[0]).flatten()[:len(feature_names)].tolist()

        shap_df = pd.DataFrame({"Feature": feature_names, "SHAP Value": patient_shap, "Patient Value": patient_vals})
        shap_df = shap_df.sort_values("SHAP Value", key=abs, ascending=True)

        fig_w, ax_w = plt.subplots(figsize=(8, 4))
        fig_w.patch.set_facecolor("#0f172a"); ax_w.set_facecolor("#1e293b")
        colors = ["#f87171" if v > 0 else "#34d399" for v in shap_df["SHAP Value"]]
        bars = ax_w.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors, alpha=0.85)
        ax_w.axvline(0, color="white", linewidth=1, linestyle="--", alpha=0.5)
        for bar, val, pv in zip(bars, shap_df["SHAP Value"], shap_df["Patient Value"]):
            ax_w.text(val + (0.003 if val >= 0 else -0.003), bar.get_y() + bar.get_height()/2,
                      f"{val:+.3f}  [val={pv:.1f}]", va="center", ha="left" if val >= 0 else "right", color="white", fontsize=8)
        ax_w.set_xlabel("SHAP Value (impact on prediction)", color="#94a3b8")
        ax_w.set_title(f"Feature Impact — {sel_model}", color="white", fontsize=11, fontweight="bold")
        ax_w.tick_params(colors="#94a3b8")
        for spine in ax_w.spines.values(): spine.set_edgecolor("#334155")
        red_patch = mpatches.Patch(color="#f87171", label="Increases risk")
        grn_patch = mpatches.Patch(color="#34d399", label="Decreases risk")
        ax_w.legend(handles=[red_patch, grn_patch], loc="lower right", facecolor="#1e293b", labelcolor="white", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_w, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Performance":
    st.markdown("## 📊 Model Performance Metrics")
    st.markdown("---")
    st.markdown("### 📋 Performance Summary Table")
    rows = [{"Model": name, "Accuracy": f"{m['Accuracy']:.4f}", "Precision": f"{m['Precision']:.4f}", "Recall": f"{m['Recall']:.4f}", "F1 Score": f"{m['F1 Score']:.4f}", "ROC-AUC": f"{m['ROC-AUC']:.4f}"} for name, m in metrics.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📈 Metric Comparison Chart")
    metric_cols = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
    fig_bar, ax_bar = plt.subplots(figsize=(11, 4))
    fig_bar.patch.set_facecolor("#0f172a"); ax_bar.set_facecolor("#1e293b")
    x = np.arange(len(metric_cols)); width = 0.15
    colors_m = ["#38bdf8","#818cf8","#f472b6","#fb7185","#a78bfa"]
    for i, (name, m) in enumerate(metrics.items()):
        vals = [m[c] for c in metric_cols]
        bars = ax_bar.bar(x + i*width - width*2, vals, width, label=name, color=colors_m[i], alpha=0.85, edgecolor="#0f172a")
        for bar, val in zip(bars, vals):
            ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color="white")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(metric_cols, color="#94a3b8")
    ax_bar.set_ylim(0, 1.12); ax_bar.set_ylabel("Score", color="#94a3b8")
    ax_bar.set_title("Model Performance Comparison", color="white", fontsize=12, fontweight="bold")
    ax_bar.tick_params(colors="#94a3b8"); ax_bar.legend(facecolor="#1e293b", labelcolor="white")
    for sp in ax_bar.spines.values(): sp.set_edgecolor("#334155")
    ax_bar.yaxis.grid(True, color="#334155", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig_bar, use_container_width=True); plt.close()

    st.markdown("---")
    st.markdown("### 🔢 Confusion Matrices")
    cm_cols = st.columns(3)
    model_items = list(metrics.items())
    for i in range(3):
        with cm_cols[i]:
            name, m = model_items[i]
            st.markdown(f"**{name}**")
            cm = np.array(m["confusion_matrix"])
            fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3))
            fig_cm.patch.set_facecolor("#0f172a"); ax_cm.set_facecolor("#1e293b")
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease","Disease"], yticklabels=["No Disease","Disease"], ax=ax_cm, cbar=False, annot_kws={"size":13,"weight":"bold","color":"black"})
            ax_cm.set_xlabel("Predicted", color="#94a3b8", fontsize=9); ax_cm.set_ylabel("Actual", color="#94a3b8", fontsize=9)
            ax_cm.tick_params(colors="#94a3b8", labelsize=7); ax_cm.set_title(name, color="white", fontsize=9, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig_cm, use_container_width=True); plt.close()
            
    cm_cols2 = st.columns(2)
    for i in range(3, 5):
        with cm_cols2[i-3]:
            name, m = model_items[i]
            st.markdown(f"**{name}**")
            cm = np.array(m["confusion_matrix"])
            fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3))
            fig_cm.patch.set_facecolor("#0f172a"); ax_cm.set_facecolor("#1e293b")
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease","Disease"], yticklabels=["No Disease","Disease"], ax=ax_cm, cbar=False, annot_kws={"size":13,"weight":"bold","color":"black"})
            ax_cm.set_xlabel("Predicted", color="#94a3b8", fontsize=9); ax_cm.set_ylabel("Actual", color="#94a3b8", fontsize=9)
            ax_cm.tick_params(colors="#94a3b8", labelsize=7); ax_cm.set_title(name, color="white", fontsize=9, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig_cm, use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ROC CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  ROC Curves":
    st.markdown("## 📈 ROC Curves & AUC Analysis")
    st.markdown("---")
    c1, c2 = st.columns([1.6, 1])
    with c1:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 5.5))
        fig_roc.patch.set_facecolor("#0f172a"); ax_roc.set_facecolor("#1e293b")
        palette = ["#38bdf8", "#818cf8", "#f472b6", "#fb7185", "#a78bfa"]
        for (name, rd), color in zip(roc_data.items(), palette):
            ax_roc.plot(rd["fpr"], rd["tpr"], lw=2.5, color=color, label=f"{name}  (AUC={rd['auc']:.3f})")
        ax_roc.plot([0,1],[0,1],"--", color="#64748b", lw=1.5, label="Random Classifier")
        ax_roc.fill_between([0,1],[0,1], alpha=0.05, color="white")
        ax_roc.set_xlabel("False Positive Rate", color="#94a3b8", fontsize=11)
        ax_roc.set_ylabel("True Positive Rate", color="#94a3b8", fontsize=11)
        ax_roc.set_title("ROC Curves — All Models", color="white", fontsize=13, fontweight="bold")
        ax_roc.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
        ax_roc.tick_params(colors="#94a3b8")
        for sp in ax_roc.spines.values(): sp.set_edgecolor("#334155")
        ax_roc.grid(True, alpha=0.2, color="#334155")
        plt.tight_layout(); st.pyplot(fig_roc, use_container_width=True); plt.close()

    with c2:
        st.markdown("### 🏅 AUC Ranking")
        auc_rows = sorted([{"Model": n, "AUC": rd["auc"]} for n, rd in roc_data.items()], key=lambda x: x["AUC"], reverse=True)
        for rank, row in enumerate(auc_rows, 1):
            medal = ["🥇","🥈","🥉","🏅","🏅"][rank-1]
            st.markdown(f'<div style="background:#1e293b; border:1px solid #334155; border-radius:10px; padding:12px 16px; margin:8px 0;"><span style="font-size:1.3rem;">{medal}</span><span style="color:#e2e8f0; font-weight:600; margin-left:8px;">{row["Model"]}</span><span style="color:#38bdf8; float:right; font-size:1.1rem; font-weight:700;">AUC = {row["AUC"]:.4f}</span></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### ℹ️ How to Read ROC")
        st.info("**ROC (Receiver Operating Characteristic)** plots the True Positive Rate vs False Positive Rate.\n- **AUC = 1.0** → Perfect model\n- **AUC = 0.5** → No better than chance\n- **Higher AUC** → Better discrimination between patients with and without heart disease")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — XAI SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠  XAI — SHAP Analysis":
    st.markdown("## 🧠 Explainable AI — SHAP Analysis")
    st.markdown("SHAP (SHapley Additive exPlanations) explains how each feature contributes to model predictions.")
    st.markdown("---")

    sel_shap_model = st.selectbox("Select Model for SHAP Analysis", MODEL_NAMES)
    sd = shap_data[sel_shap_model]
    sv = sd["shap_values_positive"]
    X_samp = sd["X_test_sample"]
    fi = sd["feature_importance"]
    is_tree_based = sd.get("is_tree_based", False)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Feature Importance", "🐝 Beeswarm Plot", "💧 Waterfall", "🔗 Dependence Plot"])

    with tab1:
        st.markdown(f"#### Global Feature Importance — {sel_shap_model}")
        
        clean_shap = []
        for val in fi["Mean_SHAP"]:
            if isinstance(val, (list, np.ndarray)):
                clean_shap.append(float(val[1]) if len(val) > 1 else float(val[0]))
            else:
                clean_shap.append(float(val))
        fi["Mean_SHAP"] = clean_shap
        fi = fi.sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)

        fig_fi, ax_fi = plt.subplots(figsize=(8, 4.5))
        fig_fi.patch.set_facecolor("#0f172a"); ax_fi.set_facecolor("#1e293b")
        bars = ax_fi.barh(fi["Feature"][::-1], fi["Mean_SHAP"][::-1], color="#38bdf8", alpha=0.85, edgecolor="#0f172a")
        for bar, val in zip(bars, fi["Mean_SHAP"][::-1]):
            ax_fi.text(val + 0.001, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va="center", color="white", fontsize=9, fontweight="bold")
        ax_fi.set_xlabel("Mean |SHAP Value|", color="#94a3b8", fontsize=10)
        ax_fi.set_title(f"SHAP Feature Importance — {sel_shap_model}", color="white", fontsize=11, fontweight="bold")
        ax_fi.tick_params(colors="#94a3b8")
        for sp in ax_fi.spines.values(): sp.set_edgecolor("#334155")
        ax_fi.xaxis.grid(True, color="#334155", alpha=0.4)
        plt.tight_layout(); st.pyplot(fig_fi, use_container_width=True); plt.close()

        c1_, c2_ = st.columns(2)
        with c1_:
            st.markdown("**Feature Importance Table**")
            st.dataframe(fi.style.background_gradient(subset=["Mean_SHAP"], cmap="Blues"), use_container_width=True, hide_index=True)
        with c2_:
            st.markdown("**Interpretation**")
            st.info(f"**{sel_shap_model} Key Insights:**\n- **{fi.iloc[0]['Feature']}** is the most influential predictor\n- **{fi.iloc[1]['Feature']}** is the second-strongest driver\n- Features with higher Mean |SHAP| push predictions further from the baseline")

    with tab2:
        st.markdown(f"#### SHAP Beeswarm / Summary Plot — {sel_shap_model}")
        st.caption("Red = high feature value; Blue = low. Points toward +x increase heart disease risk.")
        
        # FIX: Save current rcParams to restore later
        old_rcParams = plt.rcParams.copy()
        plt.rcParams.update({
            "axes.facecolor": "#1e293b",
            "text.color": "white",
            "axes.labelcolor": "#94a3b8",
            "figure.facecolor": "#0f172a",
            "xtick.color": "#94a3b8",
            "ytick.color": "#94a3b8"
        })
        
        # Ensure sv is a 2D numpy array
        sv_to_plot = sv
        if isinstance(sv_to_plot, list):
            sv_to_plot = sv_to_plot[1] if len(sv_to_plot) > 1 else sv_to_plot[0]
        sv_to_plot = np.array(sv_to_plot)
        if len(sv_to_plot.shape) == 3:
            sv_to_plot = sv_to_plot[:, :, 1]
            
        # Ensure X_samp is in the right format
        X_to_plot = X_samp.values if not is_tree_based else X_samp
        
        # FIX: Let SHAP generate the plot on the current active figure, then capture it
        shap.summary_plot(sv_to_plot, X_to_plot, plot_type="dot", feature_names=feature_names, show=False)
        
        # Capture the figure and axes SHAP just created
        fig_bee = plt.gcf()
        ax_bee = plt.gca()
        
        # Apply custom styling
        fig_bee.patch.set_facecolor("#0f172a")
        ax_bee.set_facecolor("#1e293b")
        ax_bee.tick_params(colors="#94a3b8")
        for sp in ax_bee.spines.values(): 
            sp.set_edgecolor("#334155")
            
        plt.tight_layout()
        st.pyplot(fig_bee, use_container_width=True)
        plt.close()
        
        # Restore rcParams
        plt.rcParams.update(old_rcParams)

    with tab3:
        st.markdown(f"#### Individual Patient Waterfall — {sel_shap_model}")
        patient_idx = st.slider("Select patient index from test sample", 0, min(79, len(sv)-1), 0)
        
        p_shap_raw = sv[patient_idx]
        if isinstance(p_shap_raw, (list, np.ndarray)):
            patient_shap = np.array(p_shap_raw).flatten()[:len(feature_names)].tolist()
        else:
            patient_shap = [float(p_shap_raw)] * len(feature_names)
            
        patient_data = X_samp.iloc[patient_idx]
        patient_vals = np.array(patient_data.values).flatten()[:len(feature_names)].tolist()
        ev = sd["expected_value"]

        shap_df_w = pd.DataFrame({"Feature": feature_names, "SHAP Value": patient_shap, "Patient Value": patient_vals}).sort_values("SHAP Value", key=abs, ascending=True)

        fig_wf, ax_wf = plt.subplots(figsize=(9, 4.5))
        fig_wf.patch.set_facecolor("#0f172a"); ax_wf.set_facecolor("#1e293b")
        colors_wf = ["#f87171" if v > 0 else "#34d399" for v in shap_df_w["SHAP Value"]]
        
        def format_val(v):
            try:
                return f"{float(v):.1f}"
            except (ValueError, TypeError):
                return str(v)

        bars_wf = ax_wf.barh(
            [f"{f}\n= {format_val(v)}" for f, v in zip(shap_df_w["Feature"], shap_df_w["Patient Value"])],
            shap_df_w["SHAP Value"], color=colors_wf, alpha=0.85)
            
        ax_wf.axvline(0, color="white", linewidth=1.5, linestyle="--", alpha=0.6)
        for bar, val in zip(bars_wf, shap_df_w["SHAP Value"]):
            ax_wf.text(val + (0.003 if val >= 0 else -0.003), bar.get_y()+bar.get_height()/2, f"{val:+.4f}", va="center", ha="left" if val >= 0 else "right", color="white", fontsize=8)
        ax_wf.set_xlabel("SHAP Value", color="#94a3b8")
        ax_wf.set_title(f"Waterfall — Patient #{patient_idx} | Baseline={ev:.3f}", color="white", fontsize=11, fontweight="bold")
        ax_wf.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax_wf.spines.values(): sp.set_edgecolor("#334155")
        ax_wf.xaxis.grid(True, color="#334155", alpha=0.4)
        red_p = mpatches.Patch(color="#f87171", label="Increases risk")
        grn_p = mpatches.Patch(color="#34d399", label="Decreases risk")
        ax_wf.legend(handles=[red_p, grn_p], facecolor="#1e293b", labelcolor="white", fontsize=8)
        plt.tight_layout(); st.pyplot(fig_wf, use_container_width=True); plt.close()

    with tab4:
        st.markdown(f"#### SHAP Dependence Plot — {sel_shap_model}")
        dep_feat = st.selectbox("Select feature", feature_names, index=feature_names.index("Age"))
        feat_idx = feature_names.index(dep_feat)
        
        sv_array = np.array(sv)
        if len(sv_array.shape) == 3:
            sv_array = sv_array[:, :, 1]
            
        shap_vals_feat = sv_array[:, feat_idx]
        feat_vals = X_samp[dep_feat].values

        fig_dep, ax_dep = plt.subplots(figsize=(8, 4))
        fig_dep.patch.set_facecolor("#0f172a"); ax_dep.set_facecolor("#1e293b")
        sc = ax_dep.scatter(feat_vals, shap_vals_feat, c=feat_vals, cmap="RdYlGn_r", s=40, alpha=0.75, edgecolors="none")
        ax_dep.axhline(0, color="white", linewidth=1, linestyle="--", alpha=0.5)
        cb = plt.colorbar(sc, ax=ax_dep)
        cb.set_label(dep_feat, color="#94a3b8", fontsize=9); cb.ax.tick_params(colors="#94a3b8")
        ax_dep.set_xlabel(dep_feat, color="#94a3b8", fontsize=10)
        ax_dep.set_ylabel("SHAP Value", color="#94a3b8", fontsize=10)
        ax_dep.set_title(f"Dependence: {dep_feat} → SHAP Impact | {sel_shap_model}", color="white", fontsize=11, fontweight="bold")
        ax_dep.tick_params(colors="#94a3b8")
        for sp in ax_dep.spines.values(): sp.set_edgecolor("#334155")
        ax_dep.grid(True, color="#334155", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_dep, use_container_width=True); plt.close()
        st.info(f"**Insight:** Points above y=0 indicate higher {dep_feat} values **increase** heart disease risk. Points below y=0 suggest a **protective** or neutral effect.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗂️  Data Explorer":
    st.markdown("## 🗂️ Data Explorer")
    st.markdown("Explore the test dataset used for model evaluation.")
    st.markdown("---")

    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test["Outcome"] = y_test.values
    df_test["Outcome_Label"] = df_test["Outcome"].map({0:"No Heart Disease", 1:"Heart Disease"})

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Test Patients", len(df_test))
        st.metric("Heart Disease", int(df_test["Outcome"].sum()))
    with c2:
        st.metric("No Heart Disease", int((df_test["Outcome"]==0).sum()))
        st.metric("Prevalence", f"{df_test['Outcome'].mean():.1%}")

    st.markdown("### 🔍 Filter & Browse")
    outcome_filter = st.multiselect("Filter by Outcome", ["No Heart Disease","Heart Disease"], default=["No Heart Disease","Heart Disease"])
    df_show = df_test[df_test["Outcome_Label"].isin(outcome_filter)].drop("Outcome", axis=1)
    st.dataframe(df_show, use_container_width=True, hide_index=False)

    st.markdown("---")
    st.markdown("### 📊 Feature Distribution by Outcome")
    num_features_list = ['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak', 'Ca']
    dist_feat = st.selectbox("Select Feature", feature_names)
    
    fig_dist, ax_dist = plt.subplots(figsize=(8, 3.5))
    fig_dist.patch.set_facecolor("#0f172a"); ax_dist.set_facecolor("#1e293b")
    
    if dist_feat in num_features_list:
        for label, color in [("No Heart Disease","#34d399"), ("Heart Disease","#f87171")]:
            vals = df_test[df_test["Outcome_Label"]==label][dist_feat]
            ax_dist.hist(vals, bins=25, alpha=0.65, color=color, label=label, edgecolor="#0f172a")
        ax_dist.set_ylabel("Count", color="#94a3b8")
    else:
        for label, color in [("No Heart Disease","#34d399"), ("Heart Disease","#f87171")]:
            vals = df_test[df_test["Outcome_Label"]==label][dist_feat].value_counts()
            ax_dist.bar([str(v) for v in vals.index], vals.values, alpha=0.65, color=color, label=label, edgecolor="#0f172a")
        ax_dist.set_ylabel("Frequency", color="#94a3b8")
        plt.xticks(rotation=45, color="#94a3b8")

    ax_dist.set_xlabel(dist_feat, color="#94a3b8")
    ax_dist.set_title(f"Distribution of {dist_feat} by Heart Disease Status", color="white", fontsize=11, fontweight="bold")
    ax_dist.legend(facecolor="#1e293b", labelcolor="white")
    ax_dist.tick_params(colors="#94a3b8")
    for sp in ax_dist.spines.values(): sp.set_edgecolor("#334155")
    ax_dist.yaxis.grid(True, color="#334155", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig_dist, use_container_width=True); plt.close()

    st.markdown("### 🔥 Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
    fig_corr.patch.set_facecolor("#0f172a"); ax_corr.set_facecolor("#1e293b")
    
    corr = df_test[num_features_list + ["Outcome"]].corr()
    
    mask = np.zeros_like(corr); mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, linewidths=0.5, linecolor="#0f172a", annot_kws={"size":8}, cbar_kws={"shrink":0.8})
    ax_corr.set_title("Feature Correlation Matrix (Numeric Features)", color="white", fontsize=11, fontweight="bold")
    ax_corr.tick_params(colors="#94a3b8", labelsize=8)
    plt.tight_layout(); st.pyplot(fig_corr, use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — ABOUT & METHODS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋  About & Methods":
    st.markdown("## 📋 About & Methods")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔬 Dataset")
        st.info("**Heart Disease UCI Dataset**\n- Source: Cleveland Clinic Foundation\n- Observations: 303 patients\n- Features: 13 clinical predictors\n- Target: Binary (0=No heart disease, 1=Heart disease present)\n- Prevalence: ~54% with heart disease")
        st.markdown("### ⚙️ Preprocessing")
        st.markdown("- Missing value imputation with **median** for numerical features and **most frequent** for categorical features\n- **Stratified 80/20 train/test split** (random_state=42)\n- **StandardScaler** applied to numerical features within pipelines\n- **OrdinalEncoder** used for categorical features")
        st.markdown("### 🤖 Models")
        st.markdown("| Model | Optimised Via | Key Hyperparameter |\n|---|---|---|\n| Random Forest | GridSearchCV | n_estimators, max_depth |\n| XGBoost | GridSearchCV | n_estimators, max_depth, learning_rate |\n| Logistic Regression | GridSearchCV | C (regularisation) |\n| K-Nearest Neighbours | GridSearchCV | k neighbours, weighting |\n| Support Vector Machine | GridSearchCV | kernel, C, γ |\n\nAll models tuned using **5-fold cross-validation**, scoring on **ROC-AUC**.")

    with col2:
        st.markdown("### 🧠 Explainable AI — SHAP")
        st.success("**SHAP (SHapley Additive exPlanations)**\n\nSHAP uses cooperative game theory to attribute each feature's contribution to a prediction.\n- **TreeExplainer** used for tree-based models (Random Forest, XGBoost)\n- **KernelExplainer** used for non-tree models (Logistic Regression, KNN, SVM)\n- SHAP values computed on 80 held-out test patients\n\n**Visualisations provided:**\n- Feature Importance (mean |SHAP|)\n- Beeswarm plot (global distribution)\n- Waterfall chart (individual explanation)\n- Dependence plot (feature × SHAP relationship)")
        st.markdown("### 🎯 Risk Stratification")
        st.markdown("| Range | Level | Clinical Guidance |\n|---|---|---|\n| 0–30% | 🟢 Low | Routine care & primary prevention |\n| 31–50% | 🟡 Mild | Preventive action & non-invasive testing |\n| 51–70% | 🟠 Moderate | Urgent cardiology evaluation |\n| 71–100% | 🔴 High | Immediate intervention & invasive diagnostics |")
        st.markdown("### ⚠️ Disclaimer")
        st.warning("This tool is intended for **educational and research purposes only**.\nIt should **not** replace professional medical judgment.\nPredictions should be interpreted alongside full clinical assessment.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built by Researchers From CDAM")
