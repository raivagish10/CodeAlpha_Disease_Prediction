
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

RANDOM_STATE = 42


def get_models():
    """Returns a dict of {name: sklearn-compatible estimator}."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    else:
        # Same gradient-boosted-tree family as XGBoost; used as a fallback
        # since xgboost isn't installed / can't be downloaded offline.
        models["XGBoost*"] = HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE
        )

    return models


def run_experiment(X: pd.DataFrame, y: pd.Series, dataset_name: str, test_size=0.2):
    """
    Splits data, trains every model in get_models() inside a
    StandardScaler pipeline, and returns a results dataframe plus
    per-model fitted pipelines / predictions for downstream plotting.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    results = []
    fitted = {}

    for name, model in get_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        results.append(
            {
                "Dataset": dataset_name,
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": auc,
            }
        )
        fitted[name] = {
            "pipeline": pipe,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "confusion_matrix": cm,
            "fpr": fpr,
            "tpr": tpr,
        }

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return results_df, fitted
