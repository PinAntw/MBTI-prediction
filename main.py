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
from model import get_models, get_scorers, run_cv_model, get_soft_voting_ensemble

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

# #============================= Stage 1: Data Preprocessing =====================================

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

#==================================Stage 2: Normalization & SMOTE ==================================================
# mbti_df = load_processed_data()
# feature_dict = get_feature_columns(mbti_df)

# X_train, X_test, y_train, y_test = split_data(mbti_df, 'E/I', 'bert(set2)', feature_dict, smote=False)




#==================================Stage 3:  Model Training ==================================================
# #-------------- RandomForest Part TF-IDF + 詞語統計 + 情感分析子資料集--------
# from sklearn.ensemble import RandomForestClassifier

# raw_data = {
#     'EI': (X_train_EI_1, y_train_EI),
#     'SN': (X_train_SN_1, y_train_SN),
#     'TF': (X_train_TF_1, y_train_TF),
#     'JP': (X_train_JP_1, y_train_JP)
# }

# # 1. 資料切分與標準化
# split_data = {}
# for dim, (X, y) in raw_data.items():
#     X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_tr_scaled = scaler.fit_transform(X_tr)
#     X_val_scaled = scaler.transform(X_val)
#     split_data[dim] = (X_tr_scaled, X_val_scaled, y_tr, y_val)

# # 2. 固定參數的 Random Forest 訓練與評估
# rf_params = {
#     'n_estimators': 300,
#     'min_samples_split': 10,
#     'min_samples_leaf': 1,
#     'max_depth': 20,
#     'random_state': 42,
#     'n_jobs': -1
# }

# rf_models = {}
# rf_scores = {}

# for dim in ['EI', 'SN', 'TF', 'JP']:
#     X_train, X_val, y_train, y_val = split_data[dim]
#     clf = RandomForestClassifier(**rf_params)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_val)

#     print(f"\n----- {dim} dimension -----")
#     print(classification_report(y_val, y_pred))

#     rf_models[dim] = clf
#     rf_scores[dim] = clf.score(X_val, y_val)

# #-------------- SVM Part TF-IDF + 詞語統計 + 情感分析子資料集--------

# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report

# svm_models = {}

# for dim, (X, y) in raw_data.items():
#     # 資料拆分：訓練 80%、驗證 20%
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     # 特徵標準化（SVM 對尺度非常敏感）
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)

#     # 建立 SVM 模型（Gaussian kernel）
#     clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
#     clf.fit(X_train_scaled, y_train)
#     y_pred = clf.predict(X_val_scaled)

#     print(f"\n----- {dim} dimension (SVM RBF) -----")
#     print(classification_report(y_val, y_pred))

#     svm_models[dim] = clf

# #-------------- KNN Part TF-IDF + 詞語統計 + 情感分析子資料集--------
# raw_data = {
#     'EI': (X_train_EI_1, y_train_EI),
#     'SN': (X_train_SN_1, y_train_SN),
#     'TF': (X_train_TF_1, y_train_TF),
#     'JP': (X_train_JP_1, y_train_JP)
# }
# split_data = {}

# # 儲存每個維度的結果
# search_results = {}
# # 儲存最佳模型
# knn_models = {}

# # Step 1: 拆分與標準化
# for dim, (X, y) in raw_data.items():
#     X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_tr_scaled = scaler.fit_transform(X_tr)
#     X_val_scaled = scaler.transform(X_val)
#     split_data[dim] = (X_tr_scaled, X_val_scaled, y_tr, y_val)

# # Step 2: 調參與記錄交叉驗證準確率
# param_grid = {
#     'n_neighbors': [3, 5, 7, 9],
#     'weights': ['uniform', 'distance']
# }

# for dim in tqdm(['EI', 'SN', 'TF', 'JP'], desc="KNN GridSearch"):
#     X_train, X_val, y_train, y_val = split_data[dim]
#     grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', return_train_score=True, n_jobs=-1)
#     grid.fit(X_train, y_train)

#     best_model = grid.best_estimator_
#     y_pred = best_model.predict(X_val)

#     print(f"\n----- {dim} dimension -----")
#     print("Best params:", grid.best_params_)
#     print(classification_report(y_val, y_pred))

#     # 儲存
#     knn_models[dim] = best_model
#     results_df = pd.DataFrame(grid.cv_results_)
#     search_results[dim] = results_df

# # Step 3: 繪圖顯示所有維度的 KNN 調參結果
# plt.figure(figsize=(12, 8))

# for dim, results_df in search_results.items():
#     for weight in results_df['param_weights'].unique():
#         subset = results_df[results_df['param_weights'] == weight]
#         plt.plot(
#             subset['param_n_neighbors'],
#             subset['mean_test_score'],
#             label=f'{dim} (weights={weight})'
#         )

# plt.xlabel('k (n_neighbors)', fontsize=12)
# plt.ylabel('Cross-Validation Accuracy', fontsize=12)
# plt.title('KNN Accuracy vs k (All MBTI Dimensions)', fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.show()
# #-------------- KNN Part BERT + 詞語統計 + 情感分析子資料集 --------
# raw_data = {
#     'EI': (X_train_EI_2, y_train_EI),
#     'SN': (X_train_SN_2, y_train_SN),
#     'TF': (X_train_TF_2, y_train_TF),
#     'JP': (X_train_JP_2, y_train_JP)
# }
# split_data = {}

# # 儲存每個維度的結果
# search_results = {}
# # 儲存最佳模型
# knn_models = {}

# # Step 1: 拆分與標準化
# for dim, (X, y) in raw_data.items():
#     X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_tr_scaled = scaler.fit_transform(X_tr)
#     X_val_scaled = scaler.transform(X_val)
#     split_data[dim] = (X_tr_scaled, X_val_scaled, y_tr, y_val)

# # Step 2: 調參與記錄交叉驗證準確率
# param_grid = {
#     'n_neighbors': [3, 5, 7, 9],
#     'weights': ['uniform', 'distance']
# }

# for dim in tqdm(['EI', 'SN', 'TF', 'JP'], desc="KNN GridSearch"):
#     X_train, X_val, y_train, y_val = split_data[dim]
#     grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', return_train_score=True, n_jobs=-1)
#     grid.fit(X_train, y_train)

#     best_model = grid.best_estimator_
#     y_pred = best_model.predict(X_val)

#     print(f"\n----- {dim} dimension -----")
#     print("Best params:", grid.best_params_)
#     print(classification_report(y_val, y_pred))

#     # 儲存
#     knn_models[dim] = best_model
#     results_df = pd.DataFrame(grid.cv_results_)
#     search_results[dim] = results_df

# # Step 3: 繪圖顯示所有維度的 KNN 調參結果
# plt.figure(figsize=(12, 8))

# for dim, results_df in search_results.items():
#     for weight in results_df['param_weights'].unique():
#         subset = results_df[results_df['param_weights'] == weight]
#         plt.plot(
#             subset['param_n_neighbors'],
#             subset['mean_test_score'],
#             label=f'{dim} (weights={weight})'
#         )

# plt.xlabel('k (n_neighbors)', fontsize=12)
# plt.ylabel('Cross-Validation Accuracy', fontsize=12)
# plt.title('KNN Accuracy vs k (All MBTI Dimensions)', fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.show()

# #-------------- KNN Part --------
# raw_data = {
#     'EI': (X_train_EI, y_train_EI),
#     'SN': (X_train_SN, y_train_SN),
#     'TF': (X_train_TF, y_train_TF),
#     'JP': (X_train_JP, y_train_JP)
# }
# split_data = {}

# # 儲存每個維度的結果
# search_results = {}
# # 儲存最佳模型
# knn_models = {}

# # Step 1: 拆分與標準化
# for dim, (X, y) in raw_data.items():
#     X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_tr_scaled = scaler.fit_transform(X_tr)
#     X_val_scaled = scaler.transform(X_val)
#     split_data[dim] = (X_tr_scaled, X_val_scaled, y_tr, y_val)

# # Step 2: 調參與記錄交叉驗證準確率
# param_grid = {
#     'n_neighbors': [3, 5, 7, 9],
#     'weights': ['uniform', 'distance']
# }

# for dim in tqdm(['EI', 'SN', 'TF', 'JP'], desc="KNN GridSearch"):
#     X_train, X_val, y_train, y_val = split_data[dim]
#     grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', return_train_score=True, n_jobs=-1)
#     grid.fit(X_train, y_train)

#     best_model = grid.best_estimator_
#     y_pred = best_model.predict(X_val)

#     print(f"\n----- {dim} dimension -----")
#     print("Best params:", grid.best_params_)
#     print(classification_report(y_val, y_pred))

#     # 儲存
#     knn_models[dim] = best_model
#     results_df = pd.DataFrame(grid.cv_results_)
#     search_results[dim] = results_df

# # Step 3: 繪圖顯示所有維度的 KNN 調參結果
# plt.figure(figsize=(12, 8))

# for dim, results_df in search_results.items():
#     for weight in results_df['param_weights'].unique():
#         subset = results_df[results_df['param_weights'] == weight]
#         plt.plot(
#             subset['param_n_neighbors'],
#             subset['mean_test_score'],
#             label=f'{dim} (weights={weight})'
#         )

# plt.xlabel('k (n_neighbors)', fontsize=12)
# plt.ylabel('Cross-Validation Accuracy', fontsize=12)
# plt.title('KNN Accuracy vs k (All MBTI Dimensions)', fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.show()
# #==================================Stage 4: Predict & Model Evaluation =======================================
#第一階段：特徵組合效能比較（無 SMOTE）
mbti_df = load_processed_data()
feature_dict = get_feature_columns(mbti_df)
dimensions = ['E/I', 'S/N', 'T/F', 'J/P']
feature_sets = ['tfidf(set1)', 'bert(set2)', 'all(set3)']
models = get_models()
scorers = get_scorers()

stage1_results = []

for dim in dimensions:
    for feature_set in feature_sets:
        X_train, _, y_train, _ = split_data(mbti_df, dim, feature_set, feature_dict, smote=False)
        for model_name, model in models.items():
            scores = run_cv_model(model, X_train, y_train, model_name=f"{model_name}_{dim}_{feature_set}")
            stage1_results.append({
                'Dimension': dim,
                'FeatureSet': feature_set,
                'Model': model_name,
                **scores
            })

pd.DataFrame(stage1_results).to_csv("stage1_results.csv", index=False)

# 第二階段：SMOTE 有無比較（使用 Dataset3）
# 根據第一階段結果選定 Dataset3（all(set3)）
selected_feature = 'all(set3)'
stage2_results = []

for dim in dimensions:
    for smote_flag in [False, True]:
        X_train, _, y_train, _ = split_data(mbti_df, dim, selected_feature, feature_dict, smote=smote_flag)
        for model_name in ['LogisticRegression', 'XGBoost', 'SVM_RBF']:
            model = get_models()[model_name]
            scores = run_cv_model(model, X_train, y_train, model_name=f"{model_name}_{dim}_smote={smote_flag}")
            stage2_results.append({
                'Dimension': dim,
                'Model': model_name,
                'SMOTE': smote_flag,
                **scores
            })

pd.DataFrame(stage2_results).to_csv("stage2_smote_comparison.csv", index=False)

#第三階段：Ensemble 比較
from model import get_soft_voting_ensemble

ensemble_results = []
selected_models = ['LogisticRegression', 'XGBoost', 'SVM_RBF']
model_subset = {k: get_models()[k] for k in selected_models}

for dim in dimensions:
    X_train, _, y_train, _ = split_data(mbti_df, dim, selected_feature, feature_dict, smote=True)

    # 單一模型表現
    for model_name, model in model_subset.items():
        scores = run_cv_model(model, X_train, y_train, model_name=f"{model_name}_{dim}")
        ensemble_results.append({
            'Dimension': dim,
            'Setting': model_name,
            **scores
        })

    # Soft Voting
    ensemble = get_soft_voting_ensemble(model_subset)
    scores = run_cv_model(ensemble, X_train, y_train, model_name=f"SoftVoting_{dim}")
    ensemble_results.append({
        'Dimension': dim,
        'Setting': 'SoftVoting',
        **scores
    })

pd.DataFrame(ensemble_results).to_csv("stage3_ensemble_comparison.csv", index=False)
