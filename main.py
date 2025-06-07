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
mbti_df = pd.read_csv('mbti_full.csv', encoding='utf-8', encoding_errors='ignore')

mbti_df['E/I'] = mbti_df['type'].str[0]
mbti_df['S/N'] = mbti_df['type'].str[1]
mbti_df['T/F'] = mbti_df['type'].str[2]
mbti_df['J/P'] = mbti_df['type'].str[3]

def parse_vector(s):
    s = s.strip('[]')
    return [float(x) for x in re.split(r'\s+', s.strip()) if x]

def expand_vector_column(df, col_name, prefix):
    parsed = df[col_name].apply(parse_vector)
    length = len(parsed.iloc[0])
    return pd.DataFrame(parsed.tolist(), columns=[f'{prefix}_{i}' for i in range(length)])

bert_df = expand_vector_column(mbti_df, 'bert_vector', 'bert')
minilm_df = expand_vector_column(mbti_df, 'minilm_vector', 'minilm')

mbti_df = pd.concat([mbti_df.drop(['type', 'posts', 'bert_vector', 'minilm_vector'], axis=1), bert_df, minilm_df], axis=1)

# ====== 插入前處理動作 ======

# 1. 對指定欄位進行 MinMax 縮放
need_scaling = ['anger', 'anticipation', 'disgust','fear','joy','sadness','surprise','trust']
scaler = MinMaxScaler()
mbti_df[need_scaling] = scaler.fit_transform(mbti_df[need_scaling])

# 2. 對 vader_label 做 one-hot encoding，並轉換 bool 欄位
vader_one_hot = pd.get_dummies(mbti_df['vader_label'], prefix='vader')
mbti_df = pd.concat([mbti_df.drop(columns=['vader_label']), vader_one_hot], axis=1)
bool_cols = mbti_df.select_dtypes(include='bool').columns
mbti_df[bool_cols] = mbti_df[bool_cols].astype(int)


tfidf_cols = [col for col in mbti_df.columns if col.startswith("tfidf_")]
bert_cols = [col for col in mbti_df.columns if col.startswith("bert_") or col.startswith("minilm_")]
sentiment_cols = ['vader_pos', 'vader_neu', 'vader_neg', 'vader_compound','vader_label']
linguistic_cols = [col for col in mbti_df.columns if col not in (tfidf_cols + bert_cols + sentiment_cols + ['E/I', 'S/N', 'T/F', 'J/P', 'Unnamed: 0'])]

feature_cols = tfidf_cols + bert_cols + sentiment_cols + linguistic_cols

X = mbti_df.select_dtypes(include=['number'])
y_EI = mbti_df['E/I']
y_SN = mbti_df['S/N']
y_TF = mbti_df['T/F']
y_JP = mbti_df['J/P']

def split_and_smote(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test

X_train_EI, X_test_EI, y_train_EI, y_test_EI = split_and_smote(X, y_EI)
X_train_SN, X_test_SN, y_train_SN, y_test_SN = split_and_smote(X, y_SN)
X_train_TF, X_test_TF, y_train_TF, y_test_TF = split_and_smote(X, y_TF)
X_train_JP, X_test_JP, y_train_JP, y_test_JP = split_and_smote(X, y_JP)

feature_set_1 = tfidf_cols + linguistic_cols + sentiment_cols  # TF-IDF + 詞語統計 + 情感
feature_set_2 = bert_cols + linguistic_cols + sentiment_cols   # BERT + 詞語統計 + 情感

def safe_column_subset(df, columns):
    return df[[col for col in columns if col in df.columns]]

# E/I
X_train_EI_1 = safe_column_subset(X_train_EI, feature_set_1)
X_train_EI_2 = safe_column_subset(X_train_EI, feature_set_2)
X_test_EI_1 = safe_column_subset(X_test_EI, feature_set_1)
X_test_EI_2 = safe_column_subset(X_test_EI, feature_set_2)

# S/N
X_train_SN_1 = safe_column_subset(X_train_SN, feature_set_1)
X_train_SN_2 = safe_column_subset(X_train_SN, feature_set_2)
X_test_SN_1 = safe_column_subset(X_test_SN, feature_set_1)
X_test_SN_2 = safe_column_subset(X_test_SN, feature_set_2)

# T/F
X_train_TF_1 = safe_column_subset(X_train_TF, feature_set_1)
X_train_TF_2 = safe_column_subset(X_train_TF, feature_set_2)
X_test_TF_1 = safe_column_subset(X_test_TF, feature_set_1)
X_test_TF_2 = safe_column_subset(X_test_TF, feature_set_2)

# J/P
X_train_JP_1 = safe_column_subset(X_train_JP, feature_set_1)
X_train_JP_2 = safe_column_subset(X_train_JP, feature_set_2)
X_test_JP_1 = safe_column_subset(X_test_JP, feature_set_1)
X_test_JP_2 = safe_column_subset(X_test_JP, feature_set_2)

print(f"詞語統計特徵維度：{len(linguistic_cols)}")
print(f"BERT 向量特徵維度：{len(bert_cols)}")
print(f"情感分析特徵維度：{len(sentiment_cols)}")
print(f"TF-IDF 特徵維度：{len(tfidf_cols)}")

print("特徵維度統計：")
print(f"1. TF-IDF + 詞語統計 + 情感分析：{len(feature_set_1)} 維")
print(f"2. BERT + 詞語統計 + 情感分析：{len(feature_set_2)} 維")
print(f"3. TF-IDF + BERT + 詞語統計 + 情感分析（全特徵）：{X.shape[1]} 維")

#==================================Stage 3:  Model Training ==================================================
#-------------- KNN Part TF-IDF + 詞語統計 + 情感分析子資料集--------
raw_data = {
    'EI': (X_train_EI_1, y_train_EI),
    'SN': (X_train_SN_1, y_train_SN),
    'TF': (X_train_TF_1, y_train_TF),
    'JP': (X_train_JP_1, y_train_JP)
}
split_data = {}

# 儲存每個維度的結果
search_results = {}
# 儲存最佳模型
knn_models = {}

# Step 1: 拆分與標準化
for dim, (X, y) in raw_data.items():
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    split_data[dim] = (X_tr_scaled, X_val_scaled, y_tr, y_val)

# Step 2: 調參與記錄交叉驗證準確率
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

for dim in tqdm(['EI', 'SN', 'TF', 'JP'], desc="KNN GridSearch"):
    X_train, X_val, y_train, y_val = split_data[dim]
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', return_train_score=True, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_val)

    print(f"\n----- {dim} dimension -----")
    print("Best params:", grid.best_params_)
    print(classification_report(y_val, y_pred))

    # 儲存
    knn_models[dim] = best_model
    results_df = pd.DataFrame(grid.cv_results_)
    search_results[dim] = results_df

# Step 3: 繪圖顯示所有維度的 KNN 調參結果
plt.figure(figsize=(12, 8))

for dim, results_df in search_results.items():
    for weight in results_df['param_weights'].unique():
        subset = results_df[results_df['param_weights'] == weight]
        plt.plot(
            subset['param_n_neighbors'],
            subset['mean_test_score'],
            label=f'{dim} (weights={weight})'
        )

plt.xlabel('k (n_neighbors)', fontsize=12)
plt.ylabel('Cross-Validation Accuracy', fontsize=12)
plt.title('KNN Accuracy vs k (All MBTI Dimensions)', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
#-------------- KNN Part BERT + 詞語統計 + 情感分析子資料集 --------
raw_data = {
    'EI': (X_train_EI_2, y_train_EI),
    'SN': (X_train_SN_2, y_train_SN),
    'TF': (X_train_TF_2, y_train_TF),
    'JP': (X_train_JP_2, y_train_JP)
}
split_data = {}

# 儲存每個維度的結果
search_results = {}
# 儲存最佳模型
knn_models = {}

# Step 1: 拆分與標準化
for dim, (X, y) in raw_data.items():
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    split_data[dim] = (X_tr_scaled, X_val_scaled, y_tr, y_val)

# Step 2: 調參與記錄交叉驗證準確率
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

for dim in tqdm(['EI', 'SN', 'TF', 'JP'], desc="KNN GridSearch"):
    X_train, X_val, y_train, y_val = split_data[dim]
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', return_train_score=True, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_val)

    print(f"\n----- {dim} dimension -----")
    print("Best params:", grid.best_params_)
    print(classification_report(y_val, y_pred))

    # 儲存
    knn_models[dim] = best_model
    results_df = pd.DataFrame(grid.cv_results_)
    search_results[dim] = results_df

# Step 3: 繪圖顯示所有維度的 KNN 調參結果
plt.figure(figsize=(12, 8))

for dim, results_df in search_results.items():
    for weight in results_df['param_weights'].unique():
        subset = results_df[results_df['param_weights'] == weight]
        plt.plot(
            subset['param_n_neighbors'],
            subset['mean_test_score'],
            label=f'{dim} (weights={weight})'
        )

plt.xlabel('k (n_neighbors)', fontsize=12)
plt.ylabel('Cross-Validation Accuracy', fontsize=12)
plt.title('KNN Accuracy vs k (All MBTI Dimensions)', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
#-------------- KNN Part --------
raw_data = {
    'EI': (X_train_EI, y_train_EI),
    'SN': (X_train_SN, y_train_SN),
    'TF': (X_train_TF, y_train_TF),
    'JP': (X_train_JP, y_train_JP)
}
split_data = {}

# 儲存每個維度的結果
search_results = {}
# 儲存最佳模型
knn_models = {}

# Step 1: 拆分與標準化
for dim, (X, y) in raw_data.items():
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    split_data[dim] = (X_tr_scaled, X_val_scaled, y_tr, y_val)

# Step 2: 調參與記錄交叉驗證準確率
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

for dim in tqdm(['EI', 'SN', 'TF', 'JP'], desc="KNN GridSearch"):
    X_train, X_val, y_train, y_val = split_data[dim]
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', return_train_score=True, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_val)

    print(f"\n----- {dim} dimension -----")
    print("Best params:", grid.best_params_)
    print(classification_report(y_val, y_pred))

    # 儲存
    knn_models[dim] = best_model
    results_df = pd.DataFrame(grid.cv_results_)
    search_results[dim] = results_df

# Step 3: 繪圖顯示所有維度的 KNN 調參結果
plt.figure(figsize=(12, 8))

for dim, results_df in search_results.items():
    for weight in results_df['param_weights'].unique():
        subset = results_df[results_df['param_weights'] == weight]
        plt.plot(
            subset['param_n_neighbors'],
            subset['mean_test_score'],
            label=f'{dim} (weights={weight})'
        )

plt.xlabel('k (n_neighbors)', fontsize=12)
plt.ylabel('Cross-Validation Accuracy', fontsize=12)
plt.title('KNN Accuracy vs k (All MBTI Dimensions)', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
#==================================Stage 4: Predict & Model Evaluation =======================================