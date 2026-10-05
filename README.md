# Graduate Admission Prediction

A Deep Learning regression project using Python and TensorFlow/Keras to predict student graduate admission chances ('Chance of Admit') based on academic and profile features.

## Dataset Features
- GRE Score
- TOEFL Score
- University Rating
- Statement of Purpose (SOP)
- Letter of Recommendation (LOR)
- Undergraduate CGPA
- Research Experience (0 or 1)
- Target: Chance of Admit (0 to 1)

## Project Pipeline
1. Data Loading and Cleaning (Pandas)
2. Preprocessing and Feature Scaling (scikit-learn MinMaxScaler, train-test split)
3. Model Architecture (Keras Sequential with Dense layers)
4. Compilation and Model Training (MSE loss, Adam optimizer)
5. Evaluation and Loss Visualization (MSE, R2 Score, Matplotlib)
