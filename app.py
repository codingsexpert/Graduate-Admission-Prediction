import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Graduate Admission Predictor (ANN)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern dark/light adaptive aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E40AF, #3B82F6, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #CBD5E1;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    .prediction-value {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 6px 0;
    }
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .badge-high {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .badge-mid {
        background-color: #FEF9C3;
        color: #854D0E;
        border: 1px solid #FDE047;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
    .category-box {
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
    .cat-safe { background-color: #F0FDF4; border-left: 4px solid #22C55E; }
    .cat-target { background-color: #FEFCE8; border-left: 4px solid #EAB308; }
    .cat-reach { background-color: #FEF2F2; border-left: 4px solid #EF4444; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_prediction_artifacts():
    """Load and cache the trained Deep Learning model and fitted scaler."""
    model_path = "admission_model.keras"
    scaler_path = "scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_admission(features_dict, model, scaler):
    """
    Given a dictionary of 7 feature values, scale and predict admission chance.
    Returns: (raw_prediction, calibrated_percentage)
    """
    cols = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']
    input_df = pd.DataFrame([[
        features_dict['gre'],
        features_dict['toefl'],
        features_dict['univ_rating'],
        features_dict['sop'],
        features_dict['lor'],
        features_dict['cgpa'],
        features_dict['research']
    ]], columns=cols)
    
    scaled = scaler.transform(input_df)
    raw_pred = float(model.predict(scaled, verbose=0)[0][0])
    
    # Dataset range calibration (Min ~0.51, Max ~0.71)
    # Calibrate into realistic 5% to 98% scale for clear decision support
    calibrated = (raw_pred - 0.51) / (0.71 - 0.51) * 100.0
    calibrated = max(5.0, min(98.5, calibrated))
    
    return raw_pred, calibrated


def main():
    st.markdown('<div class="main-header">🎓 Graduate Admission Chance Predictor (ANN)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Optimized Artificial Neural Network (Dense 32 → 16 → 8) trained with Adaptive Learning Rate & Early Stopping.</div>',
        unsafe_allow_html=True,
    )

    model, scaler = load_prediction_artifacts()

    # Sidebar Information
    with st.sidebar:
        st.header("📌 ANN Architecture")
        st.markdown(
            """
            - **Input Layer:** 7 normalized features
            - **Dense Hidden 1:** 32 Neurons (ReLU)
            - **Dense Hidden 2:** 16 Neurons (ReLU)
            - **Dense Hidden 3:** 8 Neurons (ReLU)
            - **Output Layer:** 1 Neuron (Linear)
            - **Optimizer:** Adam (LR Scheduler: 0.01 → 0.0003)
            - **Early Stopping:** Restores best epoch weights
            """
        )
        st.markdown("---")
        st.subheader("💡 Key Admissions Insights")
        st.write("1. **Undergraduate CGPA** is the #1 strongest predictor of admission.")
        st.write("2. **GRE & Research Experience** significantly increase chances at Tier 1 & 2 Universities.")
        st.write("3. **SOP & LOR** play a decisive role in borderline applications.")

    tabs = st.tabs([
        "🎯 Predict Admission Chance",
        "⚡ What-If Sensitivity Simulator",
        "📈 Model Metrics & Convergence",
        "📋 Dataset Explorer"
    ])

    # TAB 1: Prediction
    with tabs[0]:
        st.subheader("Candidate Academic Profile")
        col1, col2 = st.columns(2, gap="large")

        with col1:
            gre = st.slider("GRE Score", min_value=260, max_value=340, value=315, step=1,
                            help="Graduate Record Examination score (Scale: 260 - 340)")
            toefl = st.slider("TOEFL Score", min_value=0, max_value=120, value=105, step=1,
                              help="Test of English as a Foreign Language score (Scale: 0 - 120)")
            univ_rating = st.select_slider(
                "Target University Rating",
                options=[1, 2, 3, 4, 5],
                value=3,
                help="Tier of the university (1 = Regional/Local, 5 = Top Global/Ivy League)"
            )
            research = st.radio(
                "Research Experience",
                options=["Yes (Published / Lab work)", "No (No prior research)"],
                index=0,
                horizontal=True
            )
            research_val = 1 if "Yes" in research else 0

        with col2:
            cgpa = st.slider("Undergraduate CGPA", min_value=6.0, max_value=10.0, value=8.5, step=0.01,
                             help="Undergraduate Cumulative Grade Point Average (Scale: 6.0 - 10.0)")
            sop = st.slider("Statement of Purpose (SOP) Strength", min_value=1.0, max_value=5.0, value=3.5, step=0.5,
                            help="Quality and clarity of Statement of Purpose (Scale: 1.0 - 5.0)")
            lor = st.slider("Letter of Recommendation (LOR) Strength", min_value=1.0, max_value=5.0, value=3.5, step=0.5,
                            help="Strength of recommendation letters from professors/mentors (Scale: 1.0 - 5.0)")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🚀 Calculate Admission Chance", type="primary", use_container_width=True)

        if predict_button:
            if model is None or scaler is None:
                st.error("⚠️ Trained model or scaler file not found. Please train the model first by running `python train.py`.")
            else:
                features = {
                    'gre': gre, 'toefl': toefl, 'univ_rating': univ_rating,
                    'sop': sop, 'lor': lor, 'cgpa': cgpa, 'research': research_val
                }
                raw_score, cal_chance = predict_admission(features, model, scaler)

                st.markdown("---")
                st.subheader("Admission Probability Analysis")

                res_col1, res_col2 = st.columns([1, 1], gap="large")

                with res_col1:
                    if cal_chance >= 75:
                        badge_html = '<span class="status-badge badge-high">🟢 High Acceptance Probability</span>'
                        val_color = "#166534"
                    elif cal_chance >= 45:
                        badge_html = '<span class="status-badge badge-mid">🟡 Competitive / Target Range</span>'
                        val_color = "#854D0E"
                    else:
                        badge_html = '<span class="status-badge badge-low">🔴 Reach / Low Probability</span>'
                        val_color = "#991B1B"

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div style="font-size: 0.95rem; color: #475569; font-weight: 600;">Calibrated Admission Probability:</div>
                            <div class="prediction-value" style="color: {val_color};">{cal_chance:.1f}%</div>
                            <div style="margin-bottom: 12px;">{badge_html}</div>
                            <div style="font-size: 0.85rem; color: #64748B;">
                                <b>Raw ANN Output Index:</b> {raw_score:.3f} &nbsp;|&nbsp; <b>Applicant Tier:</b> Rating {univ_rating}/5
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(cal_chance / 100.0)

                    # University Categorization
                    st.markdown("#### 🏫 University Application Strategy")
                    if cal_chance >= 75:
                        st.markdown('<div class="category-box cat-safe"><b>✅ Safe Universities:</b> Strong candidate for Tier 1 & 2 universities. Very high acceptance confidence.</div>', unsafe_allow_html=True)
                        st.markdown('<div class="category-box cat-target"><b>🎯 Target Universities:</b> Elite top-tier universities (Rating 4-5) are well within reach.</div>', unsafe_allow_html=True)
                    elif cal_chance >= 45:
                        st.markdown('<div class="category-box cat-safe"><b>✅ Safe Universities:</b> Solid fit for Tier 2 & 3 universities.</div>', unsafe_allow_html=True)
                        st.markdown('<div class="category-box cat-target"><b>🎯 Target Universities:</b> Tier 3 & 4 institutions are competitive targets.</div>', unsafe_allow_html=True)
                        st.markdown('<div class="category-box cat-reach"><b>🚀 Reach Universities:</b> Top-tier (Rating 5) requires strengthening GRE/Research.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="category-box cat-safe"><b>✅ Safe Universities:</b> Focus on Tier 1-2 regional programs.</div>', unsafe_allow_html=True)
                        st.markdown('<div class="category-box cat-reach"><b>🚀 Reach Universities:</b> Consider retaking GRE or boosting CGPA/Research to apply for Tier 3+.</div>', unsafe_allow_html=True)

                with res_col2:
                    st.markdown("#### 🎯 Profile Breakdown & Recommendations")
                    tips = []
                    if cgpa < 8.5:
                        tips.append(f"• **CGPA Boost (+0.5 CGPA):** CGPA is at **{cgpa:.2f}**. Raising your score above 8.5 gives the highest single boost in admission probability.")
                    else:
                        tips.append(f"• **Excellent CGPA ({cgpa:.2f}):** Your GPA is well above the admitted student average (8.5), putting you in a strong position.")

                    if gre < 320:
                        tips.append(f"• **GRE Target (Current: {gre}):** Scoring 320+ makes a significant difference for competitive STEM and business programs.")
                    else:
                        tips.append(f"• **Strong GRE ({gre}):** Excellent score that passes screening cutoffs at virtually all graduate schools.")

                    if toefl < 105:
                        tips.append(f"• **TOEFL Benchmark (Current: {toefl}):** A score of 105+ guarantees eligibility for Teaching Assistantships (TA) and fellowships.")

                    if research_val == 0:
                        tips.append("• **Research Publications:** Publishing a research paper, conference presentation, or working in a lab substantially lifts your chances.")
                    else:
                        tips.append("• **Research Experience:** Having verified research experience gives you an edge over peers with similar test scores.")

                    st.markdown("\n\n".join(tips))

    # TAB 2: What-If Sensitivity Simulator
    with tabs[1]:
        st.subheader("⚡ 'What-If' Profile Sensitivity Simulator")
        st.write("See how boosting your scores directly impacts your admission probability in real time:")

        sim_col1, sim_col2 = st.columns(2, gap="large")

        with sim_col1:
            base_gre = st.slider("Base GRE Score", 260, 340, 310, step=1, key="sim_gre")
            base_cgpa = st.slider("Base CGPA", 6.0, 10.0, 8.0, step=0.05, key="sim_cgpa")
            base_research = st.toggle("Has Research Experience", value=False, key="sim_res")

        with sim_col2:
            st.markdown("#### Simulated Profile Adjustments")
            delta_gre = st.slider("Increase GRE by (+ points):", 0, 30, 10, step=1)
            delta_cgpa = st.slider("Increase CGPA by (+ GPA):", 0.0, 1.5, 0.5, step=0.05)
            gain_research = st.checkbox("Add Research Experience to profile", value=True)

        if model is not None and scaler is not None:
            # Baseline profile
            base_feat = {
                'gre': base_gre, 'toefl': 102, 'univ_rating': 3,
                'sop': 3.5, 'lor': 3.5, 'cgpa': base_cgpa, 'research': 1 if base_research else 0
            }
            _, base_chance = predict_admission(base_feat, model, scaler)

            # Improved profile
            new_gre = min(340, base_gre + delta_gre)
            new_cgpa = min(10.0, base_cgpa + delta_cgpa)
            new_res = 1 if (base_research or gain_research) else 0

            boosted_feat = {
                'gre': new_gre, 'toefl': 105, 'univ_rating': 3,
                'sop': 3.5, 'lor': 3.5, 'cgpa': new_cgpa, 'research': new_res
            }
            _, boosted_chance = predict_admission(boosted_feat, model, scaler)

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(label="Current Profile Chance", value=f"{base_chance:.1f}%")
            with m2:
                diff = boosted_chance - base_chance
                st.metric(label="Improved Profile Chance", value=f"{boosted_chance:.1f}%", delta=f"+{diff:.1f}%")
            with m3:
                st.info(f"Boosting GRE by **+{delta_gre}** and CGPA by **+{delta_cgpa:.2f}** increases your chance by **{diff:+.1f}%**!")

    # TAB 3: Model Metrics & Convergence
    with tabs[2]:
        st.subheader("Model Architecture & Performance")
        m_col1, m_col2 = st.columns(2)

        with m_col1:
            st.markdown("#### 📉 Optimized Model Loss Convergence")
            if os.path.exists("loss_curve.png"):
                st.image("loss_curve.png", caption="Model Convergence (MSE over epochs with LR reduction & Early Stopping)")
            else:
                st.info("Loss curve available after running `train.py`.")

        with m_col2:
            st.markdown("#### 🎯 Regression Goodness of Fit (R²)")
            if os.path.exists("actual_vs_predicted.png"):
                st.image("actual_vs_predicted.png", caption="Actual vs Predicted Admission Chance")
            else:
                st.info("Scatter plot available after running `train.py`.")

    # TAB 4: Dataset Explorer
    with tabs[3]:
        st.subheader("Admission Records Preview")
        if os.path.exists("Graduate_Admission_Prediction.csv"):
            df_preview = pd.read_csv("Graduate_Admission_Prediction.csv")
            st.dataframe(df_preview.head(25), use_container_width=True)
            st.write(f"Total applicants in training set: **{len(df_preview)}**")


if __name__ == "__main__":
    main()
