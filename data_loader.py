

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

RANDOM_STATE = 42


def load_breast_cancer_data():
    """Real Wisconsin Breast Cancer dataset (30 features, binary target)."""
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]  # 1 = benign, 0 = malignant (sklearn convention)
    return X, y


def load_heart_disease(n_samples=1000, random_state=RANDOM_STATE):
    """
    Synthetic stand-in for the UCI Cleveland Heart Disease dataset.
    Same 13 input features and value ranges as the real dataset.
    """
    rng = np.random.default_rng(random_state)

    age = rng.integers(29, 78, n_samples)
    sex = rng.integers(0, 2, n_samples)  # 1 = male, 0 = female
    cp = rng.integers(0, 4, n_samples)  # chest pain type (0-3)
    trestbps = rng.integers(94, 201, n_samples)  # resting blood pressure
    chol = rng.integers(126, 565, n_samples)  # serum cholesterol
    fbs = rng.binomial(1, 0.15, n_samples)  # fasting blood sugar > 120 mg/dl
    restecg = rng.integers(0, 3, n_samples)  # resting ECG results
    thalach = rng.integers(71, 203, n_samples)  # max heart rate achieved
    exang = rng.binomial(1, 0.33, n_samples)  # exercise induced angina
    oldpeak = np.round(rng.uniform(0, 6.2, n_samples), 1)  # ST depression
    slope = rng.integers(0, 3, n_samples)  # slope of peak exercise ST segment
    ca = rng.integers(0, 4, n_samples)  # number of major vessels colored
    thal = rng.choice([1, 2, 3], n_samples)  # thalassemia

    # Build target with realistic feature -> risk relationships (not random)
    risk_score = (
        0.03 * (age - 54)
        + 0.6 * sex
        + 0.4 * (cp >= 2)
        + 0.01 * (trestbps - 130)
        + 0.005 * (chol - 240)
        + 0.3 * fbs
        - 0.02 * (thalach - 150)
        + 0.7 * exang
        + 0.35 * oldpeak
        + 0.3 * (slope == 2)
        + 0.5 * ca
        + 0.4 * (thal == 3)
        + rng.normal(0, 1.0, n_samples)  # noise
    )
    target = (risk_score > np.percentile(risk_score, 55)).astype(int)

    X = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
    )
    y = pd.Series(target, name="target")
    return X, y


def load_diabetes_data(n_samples=1000, random_state=RANDOM_STATE):
    """
    Synthetic stand-in for the Pima Indians Diabetes dataset.
    Same 8 input features and value ranges as the real dataset.
    """
    rng = np.random.default_rng(random_state)

    pregnancies = rng.integers(0, 17, n_samples)
    glucose = rng.integers(56, 199, n_samples)
    blood_pressure = rng.integers(24, 122, n_samples)
    skin_thickness = rng.integers(7, 99, n_samples)
    insulin = rng.integers(15, 846, n_samples)
    bmi = np.round(rng.uniform(18.2, 67.1, n_samples), 1)
    dpf = np.round(rng.uniform(0.078, 2.42, n_samples), 3)  # diabetes pedigree function
    age = rng.integers(21, 81, n_samples)

    risk_score = (
        0.03 * (glucose - 120)
        + 0.02 * (bmi - 32)
        + 0.4 * dpf
        + 0.02 * (age - 33)
        + 0.01 * (blood_pressure - 70)
        + 0.05 * pregnancies
        + rng.normal(0, 1.2, n_samples)
    )
    target = (risk_score > np.percentile(risk_score, 65)).astype(int)

    X = pd.DataFrame(
        {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }
    )
    y = pd.Series(target, name="Outcome")
    return X, y


DATASETS = {
    "breast_cancer": {"loader": load_breast_cancer_data, "is_real": True},
    "heart_disease": {"loader": load_heart_disease, "is_real": False},
    "diabetes": {"loader": load_diabetes_data, "is_real": False},
}
