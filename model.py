# model.py

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, accuracy_score, roc_auc_score, recall_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score
)
def get_models():
    rf_params = {
        'n_estimators': 200,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_depth': 20,
        'random_state': 42
    }
    xgb_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.6,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    models = {
        'RandomForest': RandomForestClassifier(**rf_params),
        'XGBoost': XGBClassifier(**xgb_params),
        'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=42)
    }
    return models

def get_scorers():
    return {
        'F1': make_scorer(f1_score, average='macro'),
        'Accuracy': make_scorer(accuracy_score),
        'Recall': make_scorer(recall_score, average='macro'),
        'ROC_AUC': make_scorer(roc_auc_score, average='macro', multi_class='ovo')
    }

def run_cv_model(model, X, y, model_name, scoring_metrics=None, cv_folds=2):
    if scoring_metrics is None:
        scoring_metrics = get_scorers()

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    print(f"模型：{model_name}，使用 {cv_folds}-fold 交叉驗證")

    scores = {}
    for metric_name, scorer in scoring_metrics.items():
        try:
            cv_score = cross_val_score(model, X, y, cv=cv, scoring=scorer)
            scores[metric_name] = np.mean(cv_score)
        except Exception as e:
            scores[metric_name] = np.nan
            print(f"無法評估 {model_name} 的 {metric_name}：{e}")

    return scores

def get_soft_voting_ensemble(models: dict):
    return VotingClassifier(
        estimators=[(name, clf) for name, clf in models.items()],
        voting='soft'
    )
def evaluate_on_test(model, X_test, y_test):
    y_pred = model.predict(X_test)

    results = {
        'F1': f1_score(y_test, y_pred, average='macro'),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='macro'),
    }

    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                results['ROC_AUC'] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                results['ROC_AUC'] = roc_auc_score(y_test, y_proba, multi_class='ovo', average='macro')
        else:
            results['ROC_AUC'] = np.nan
    except Exception as e:
        results['ROC_AUC'] = np.nan
        print(f"無法計算 ROC_AUC：{e}")

    return results