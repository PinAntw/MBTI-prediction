import json
import zipfile
import os
import re
from collections import defaultdict, Counter
import requests

import pandas as pd
import matplotlib.pyplot as plt
import nltk

import statistics
from statistics import variance

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from linguisticPreprocessor import LinguisticPreprocessor
from sentimentPreprocessor import SentimentPreprocessor
from tfidfPreprocessor import TfidfPreprocessor

#pip install vaderSentiment
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from data_utils import load_processed_data, get_feature_columns, split_data
from model import get_models, get_scorers, run_cv_model, get_soft_voting_ensemble, evaluate_on_test

# #============================= Stage 0: Data Preparation =====================================
# zip_path = "mbti-type.zip"
# target_file = "mbti_1.csv"

# # 如果已經存在 csv 就不解壓
# if os.path.exists(target_file):
#     print(f"已偵測到 {target_file}，略過解壓縮。")
# else:
#     if os.path.exists(zip_path):
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall()
#         print(f"解壓縮完成：{zip_path}")
#     else:
#         print(f"找不到 zip 檔案：{zip_path}")

# #============================= Data Preprocessing =====================================

# df = pd.read_csv("mbti_1.csv")
# print(df.head())

# #-------------- 高Part Start: 訓練集詞語統計前處理 （如果已經產生新資料集就不用重複執行） ----------
# lp = LinguisticPreprocessor(nltk_data_path="./nltk_data")
# lp.setup_nltk()
# lp.load_nrc_lexicon("./OneFilePerEmotion")
# df_ling = lp.generate_features(df, text_column="posts")
# # print(df_withSentiment.head())

# #-------------- 蔣Part Start：訓練集情感前處理（如果已經產生新資料集就不用重複執行）--------
# sp = SentimentPreprocessor()
# df_ling_sent = sp.transform_dataframe(df_ling, text_column="posts")

# #-------------- 龍Part Start ：訓練集tf-idf前處理（如果已經產生新資料集就不用重複執行）--------

# tp = TfidfPreprocessor(max_features=1000)
# df_tfidf = tp.generate_features(df)
# columns_to_remove = ['tfidf_10', 'tfidf_100', 'tfidf_12', 'tfidf_15', 'tfidf_20']
# df_tfidf.drop(columns=columns_to_remove, errors='ignore', inplace=True)

# # #-------------- 合併Part ----------------------------------------------------------------
# df_ling_sent = df_ling_sent.reset_index(drop=True)
# df_tfidf = df_tfidf.reset_index(drop=True)

# if 'mbti_label' in df_tfidf.columns:
#     df_tfidf = df_tfidf.drop(columns=['mbti_label'])

# mbti_full = pd.concat([df_ling_sent, df_tfidf], axis=1)

# if "type.1" in mbti_full.columns:
#     mbti_full.drop("type.1", axis=1, inplace=True)

# mbti_full.to_csv("mbti_full.csv", index=False)
# print("已生成mbti_full.csv")

#================================== Normalization & SMOTE , Model Training, Evaluation, Prediction==================================================
mbti_df = load_processed_data()
feature_dict = get_feature_columns(mbti_df)
dimensions = ['E/I', 'S/N', 'T/F', 'J/P']
feature_sets = ['tfidf(set1)', 'bert(set2)', 'all(set3)']
models = get_models()
scorers = get_scorers()

# # 第一階段：特徵組合效能比較（無 SMOTE）
stage1_results = []
for dim in dimensions:
    for feature_set in feature_sets:
        X_train, X_test, y_train, y_test = split_data(mbti_df, dim, feature_set, feature_dict, smote=False)
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            cv_scores = run_cv_model(model, X_train, y_train, model_name=f"{model_name}_{dim}_{feature_set}")
            test_scores = evaluate_on_test(model, X_test, y_test)
            stage1_results.append({
                'Dimension': dim,
                'FeatureSet': feature_set,
                'Model': model_name,
                **{f"CV_{k}": v for k, v in cv_scores.items()},
                **{f"Test_{k}": v for k, v in test_scores.items()}
            })
pd.DataFrame(stage1_results).to_csv("stage1_results.csv", index=False)

# 第二階段：SMOTE 有無比較（使用 Dataset3）
print("開始第二階段：SMOTE 有無比較")
selected_feature = 'all(set3)'
stage2_results = []
for dim in dimensions:
    for smote_flag in [False, True]:
        X_train, X_test, y_train, y_test = split_data(mbti_df, dim, selected_feature, feature_dict, smote=smote_flag)
        for model_name in ['RandomForest', 'XGBoost', 'SVM_RBF']:
            model = get_models()[model_name]
            model.fit(X_train, y_train)
            cv_scores = run_cv_model(model, X_train, y_train, model_name=f"{model_name}_{dim}_smote={smote_flag}")
            test_scores = evaluate_on_test(model, X_test, y_test)
            stage2_results.append({
                'Dimension': dim,
                'Model': model_name,
                'SMOTE': smote_flag,
                **{f"CV_{k}": v for k, v in cv_scores.items()},
                **{f"Test_{k}": v for k, v in test_scores.items()}
            })
pd.DataFrame(stage2_results).to_csv("stage2_smote_comparison.csv", index=False)

# 第三階段：Ensemble 比較
from model import get_soft_voting_ensemble
print("開始第三階段：Ensemble 比較")
ensemble_results = []
selected_models = ['RandomForest', 'XGBoost', 'SVM_RBF']
model_subset = {k: get_models()[k] for k in selected_models}

selected_models_soft = ['RandomForest', 'XGBoost']
model_subset_soft = {k: get_models()[k] for k in selected_models_soft}
for dim in dimensions:
    X_train, X_test, y_train, y_test = split_data(mbti_df, dim, selected_feature, feature_dict, smote=True)

    # 單一模型表現
    for model_name, model in model_subset.items():
        model.fit(X_train, y_train)
        cv_scores = run_cv_model(model, X_train, y_train, model_name=f"{model_name}_{dim}")
        test_scores = evaluate_on_test(model, X_test, y_test)
        ensemble_results.append({
            'Dimension': dim,
            'Setting': model_name,
            **{f"CV_{k}": v for k, v in cv_scores.items()},
            **{f"Test_{k}": v for k, v in test_scores.items()}
        })

    # Soft Voting
    ensemble = get_soft_voting_ensemble(model_subset_soft)
    ensemble.fit(X_train, y_train)
    cv_scores = run_cv_model(ensemble, X_train, y_train, model_name=f"SoftVoting_{dim}")
    test_scores = evaluate_on_test(ensemble, X_test, y_test)
    ensemble_results.append({
        'Dimension': dim,
        'Setting': 'SoftVoting',
        **{f"CV_{k}": v for k, v in cv_scores.items()},
        **{f"Test_{k}": v for k, v in test_scores.items()}
    })
pd.DataFrame(ensemble_results).to_csv("stage3_ensemble_comparison.csv", index=False)
