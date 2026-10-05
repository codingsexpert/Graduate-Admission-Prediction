# 🎓 Graduate Admission Prediction (Deep Learning & Streamlit)

An end-to-end Deep Learning regression application built with **TensorFlow / Keras** and **Streamlit** to estimate a student's probability of admission ('Chance of Admit') based on academic credentials and profile strength.

---

## 📌 Dataset Features

The model uses 7 key academic factors:
1. **GRE Score**: Graduate Record Examination (Scale: 260 - 340)
2. **TOEFL Score**: Test of English as a Foreign Language (Scale: 0 - 120)
3. **University Rating**: Undergraduate institution tier/reputation (Scale: 1 - 5)
4. **Statement of Purpose (SOP)**: Strength of applicant SOP (Scale: 1.0 - 5.0)
5. **Letter of Recommendation (LOR)**: Strength of recommendation letters (Scale: 1.0 - 5.0)
6. **Undergraduate CGPA**: Cumulative Grade Point Average (Scale: 6.0 - 10.0)
7. **Research Experience**: Prior research/publication background (0 = No, 1 = Yes)

**Target Variable**:
- **Chance of Admit**: Continuous probability score ranging from 0.0 to 1.0 (or 0% to 100%).

---

## 🏗️ Architecture & Pipeline

1. **Data Cleaning & Loading**: Strips trailing spaces from columns and drops `Serial No.`.
2. **Preprocessing**: 
   - 80/20 train-test split (`random_state=42`).
   - Normalization using `MinMaxScaler` fit strictly on training features to prevent data leakage.
3. **Deep Learning Model (Sequential Neural Network)**:
   - `Input Layer`: 7 features
   - `Dense Layer 1`: 16 neurons with `ReLU` activation
   - `Dense Layer 2`: 8 neurons with `ReLU` activation
   - `Output Layer`: 1 neuron with `Linear` activation
4. **Optimization & Training**:
   - Optimizer: `Adam`
   - Loss Function: `Mean Squared Error (MSE)`
   - Metric: `Mean Absolute Error (MAE)`
   - Trained for 100 epochs with 20% validation split.
5. **Artifacts Saved**:
   - `admission_model.keras`: Trained Keras model.
   - `scaler.pkl`: Fitted `MinMaxScaler` object.
   - `loss_curve.png`: Training vs Validation loss curve.
   - `actual_vs_predicted.png`: Regression goodness-of-fit scatter plot.

---

## 🚀 How to Run

### 1. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Deep Learning Model
```bash
python train.py
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test custom student profiles and view interactive recommendations!

