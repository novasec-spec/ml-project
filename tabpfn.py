from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

# Load data
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------
# TabPFN
# ------------------
tabpfn = TabPFNClassifier()

tabpfn.fit(X_train, y_train)

tabpfn_preds = tabpfn.predict(X_test)
tabpfn_probs = tabpfn.predict_proba(X_test)[:, 1]

print("TabPFN Accuracy:",
      accuracy_score(y_test, tabpfn_preds))

print("TabPFN ROC-AUC:",
      roc_auc_score(y_test, tabpfn_probs))

# ------------------
# XGBoost
# ------------------
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_preds = xgb.predict(X_test)
xgb_probs = xgb.predict_proba(X_test)[:, 1]

print("XGBoost Accuracy:",
      accuracy_score(y_test, xgb_preds))

print("XGBoost ROC-AUC:",
      roc_auc_score(y_test, xgb_probs))