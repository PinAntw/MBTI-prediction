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
from linguisticPreprocessor import LinguisticPreprocessor

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from imblearn.over_sampling import SMOTE

#============================= Stage 0: Data Preparation =====================================
zip_path = "mbti-type.zip"
target_file = "mbti_1.csv"

# 如果已經存在 csv 就不解壓
if os.path.exists(target_file):
    print(f"已偵測到 {target_file}，略過解壓縮。")
else:
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        print(f"解壓縮完成：{zip_path}")
    else:
        print(f"找不到 zip 檔案：{zip_path}")

#============================= Stage 1: Data Preprocessing =====================================


#-------------- 高Part Start: 訓練集情感前處理 （如果已經產生新資料集就不用重複執行） ----------
# lp = LinguisticPreprocessor(nltk_data_path="./nltk_data")
# lp.setup_nltk()
# lp.load_nrc_lexicon("./OneFilePerEmotion")

# df = pd.read_csv("mbti_1.csv")
# df_withSentiment = lp.generate_features(df, text_column="posts")
# print(df_withSentiment.head())

# df_withSentiment.to_csv("mbti_1_with_sentiment.csv", index=False)

df = pd.read_csv("mbti_1_with_sentiment.csv")
print(df.head())

#-------------- 蔣Part Start --------


#-------------- 龍Part Start --------



mbti_df = pd.read_csv('mbti_1_with_sentiment.csv', encoding='utf-8', encoding_errors='ignore')
#==================================Stage 2: Normalization & SMOTE ==================================================
mbti_df['E/I'] = mbti_df['type'].str[0]
mbti_df['S/N'] = mbti_df['type'].str[1]
mbti_df['T/F'] = mbti_df['type'].str[2]
mbti_df['J/P'] = mbti_df['type'].str[3]
mbti_df = mbti_df.drop(['type', 'posts'], axis=1)


# feature: 所有欄位除了 MBTI 四維
feature_cols = [col for col in mbti_df.columns if col not in ['E/I', 'S/N', 'T/F', 'J/P', 'Unnamed: 0']]
X = mbti_df[feature_cols]
y_EI = mbti_df['E/I']
y_SN = mbti_df['S/N']
y_TF = mbti_df['T/F']
y_JP = mbti_df['J/P']

# === 拆分資料（80% train / 20% test）===
X_train_EI, X_test_EI, y_train_EI, y_test_EI = train_test_split(
    X, y_EI, test_size=0.2, random_state=42, stratify=y_EI
)
X_train_SN, X_test_SN, y_train_SN, y_test_SN = train_test_split(
    X, y_SN, test_size=0.2, random_state=42, stratify=y_SN
)
X_train_TF, X_test_TF, y_train_TF, y_test_TF = train_test_split(
    X, y_TF, test_size=0.2, random_state=42, stratify=y_TF
)
X_train_JP, X_test_JP, y_train_JP, y_test_JP = train_test_split(
    X, y_JP, test_size=0.2, random_state=42, stratify=y_JP
)


# 使用 SMOTE 進行上採樣
smote = SMOTE(random_state=42)
X_train_EI, y_train_EI = smote.fit_resample(X_train_EI, y_train_EI)
X_train_SN, y_train_SN = smote.fit_resample(X_train_SN, y_train_SN)
X_train_TF, y_train_TF = smote.fit_resample(X_train_TF, y_train_TF)
X_train_JP, y_train_JP = smote.fit_resample(X_train_JP, y_train_JP)


#==================================Stage 3:  Model Training ==================================================
#-------------- KNN Part --------



#==================================Stage 4: Predict & Model Evaluation =======================================