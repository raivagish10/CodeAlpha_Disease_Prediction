# Disease Prediction — Results Report

Boosting model used: HistGradientBoosting (XGBoost fallback — xgboost package unavailable offline)


## Breast Cancer  
*REAL data (UCI / sklearn)*  

Samples: 569, Features: 30, Positive class rate: 62.74%

| Dataset       | Model               |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |
|:--------------|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Breast Cancer | Logistic Regression |     0.9825 |      0.9861 |   0.9861 |     0.9861 |    0.9954 |
| Breast Cancer | SVM (RBF kernel)    |     0.9825 |      0.9861 |   0.9861 |     0.9861 |    0.995  |
| Breast Cancer | Random Forest       |     0.9474 |      0.9583 |   0.9583 |     0.9583 |    0.9937 |
| Breast Cancer | XGBoost*            |     0.9561 |      0.9467 |   0.9861 |     0.966  |    0.9921 |

**Best model:** Logistic Regression (ROC-AUC = 0.9954)


![](Breast Cancer_comparison.png)

![](Breast Cancer_confusion.png)

![](Breast Cancer_roc.png)


## Heart Disease  
*SYNTHETIC data (schema-matched to UCI dataset)*  

Samples: 1000, Features: 13, Positive class rate: 45.00%

| Dataset       | Model               |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |
|:--------------|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Heart Disease | Logistic Regression |      0.82  |      0.8293 |   0.7556 |     0.7907 |    0.897  |
| Heart Disease | SVM (RBF kernel)    |      0.815 |      0.8046 |   0.7778 |     0.791  |    0.8849 |
| Heart Disease | XGBoost*            |      0.76  |      0.7442 |   0.7111 |     0.7273 |    0.832  |
| Heart Disease | Random Forest       |      0.77  |      0.7558 |   0.7222 |     0.7386 |    0.8231 |

**Best model:** Logistic Regression (ROC-AUC = 0.8970)


![](Heart Disease_comparison.png)

![](Heart Disease_confusion.png)

![](Heart Disease_roc.png)


## Diabetes  
*SYNTHETIC data (schema-matched to UCI dataset)*  

Samples: 1000, Features: 8, Positive class rate: 35.00%

| Dataset   | Model               |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |
|:----------|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Diabetes  | Logistic Regression |      0.8   |      0.7419 |   0.6571 |     0.697  |    0.8964 |
| Diabetes  | SVM (RBF kernel)    |      0.805 |      0.7818 |   0.6143 |     0.688  |    0.8873 |
| Diabetes  | XGBoost*            |      0.815 |      0.7463 |   0.7143 |     0.7299 |    0.8857 |
| Diabetes  | Random Forest       |      0.805 |      0.7719 |   0.6286 |     0.6929 |    0.8837 |

**Best model:** Logistic Regression (ROC-AUC = 0.8964)


![](Diabetes_comparison.png)

![](Diabetes_confusion.png)

![](Diabetes_roc.png)
