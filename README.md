# Graduate Admission Prediction (Deep Learning and Streamlit)

An end-to-end Deep Learning regression application built with **TensorFlow / Keras** and **Streamlit** to estimate an applicant's probability of admission ('Chance of Admit') based on academic credentials and profile strength.

---

## Dataset Features

The model uses 7 key academic factors:
1. **GRE Score**: Graduate Record Examination (Scale: 260 - 340)
2. **TOEFL Score**: Test of English as a Foreign Language (Scale: 0 - 120)
3. **University Rating**: Undergraduate institution tier classification (Scale: 1 - 5)
4. **Statement of Purpose (SOP)**: Strength of applicant SOP (Scale: 1.0 - 5.0)
5. **Letter of Recommendation (LOR)**: Strength of recommendation letters (Scale: 1.0 - 5.0)
6. **Undergraduate CGPA**: Cumulative Grade Point Average (Scale: 6.0 - 10.0)
7. **Research Experience**: Prior research/publication background (0 = No, 1 = Yes)

**Target Variable**:
- **Chance of Admit**: Continuous score ranging from 0.0 to 1.0 (calibrated to 5% to 98% in the user interface).

---

## Architecture and Pipeline

1. **Data Cleaning and Loading**: Strips trailing whitespaces from column headers and removes irrelevant identifier columns (`Serial No.`).
2. **Preprocessing**: 
   - 80/20 train-test split (`random_state=42`).
   - Feature normalization using `MinMaxScaler` fitted exclusively on training data to prevent data leakage.
3. **Deep Learning Model (Sequential Neural Network)**:
   - `Input Layer`: 7 normalized input features
   - `Dense Layer 1`: 32 neurons with `ReLU` activation
   - `Dense Layer 2`: 16 neurons with `ReLU` activation
   - `Dense Layer 3`: 8 neurons with `ReLU` activation
   - `Output Layer`: 1 neuron with `Linear` activation
4. **Optimization and Training**:
   - Optimizer: `Adam` with learning rate scheduling (`ReduceLROnPlateau`)
   - Loss Function: `Mean Squared Error (MSE)`
   - Metric: `Mean Absolute Error (MAE)`
   - Regularization: `EarlyStopping` with automated best-weights restoration.
5. **Artifacts Generated**:
   - `admission_model.keras`: Serialized Keras neural network model.
   - `scaler.pkl`: Fitted `MinMaxScaler` object.
   - `loss_curve.png`: Training versus validation loss convergence plot.
   - `actual_vs_predicted.png`: Regression goodness-of-fit scatter plot.

---

## How to Run

### 1. Set Up Virtual Environment and Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Deep Learning Model (Optional)
```bash
python train.py
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test custom student profiles and evaluate strategic university recommendations.
