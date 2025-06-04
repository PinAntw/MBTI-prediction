import json
import zipfile
import os
import re
from collections import defaultdict, Counter
import requests

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import nltk

import statistics
from statistics import variance
from linguisticPreprocessor import LinguisticPreprocessor

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
lp = LinguisticPreprocessor(nltk_data_path="./nltk_data")
lp.setup_nltk()
lp.load_nrc_lexicon("./OneFilePerEmotion")

df = pd.read_csv("mbti_1.csv")
df_withSentiment = lp.generate_features(df, text_column="posts")
print(df_withSentiment.head())

df_withSentiment.to_csv("mbti_1_with_sentiment.csv", index=False)

df = pd.read_csv("mbti_1_with_sentiment.csv")
print(df.head())

#-------------- 蔣Part Start --------


#-------------- 龍Part Start --------



#==================================Stage 2:  SMOTE ==================================================

#==================================Stage 3:  Model Training ==================================================

#==================================Stage 4: Predict & Model Evaluation =======================================