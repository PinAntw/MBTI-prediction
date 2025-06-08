# data_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

"""
載入並處理 mbti_full.csv

統一管理特徵欄位群組（TF-IDF、BERT、ALL）

提供 split_data(...) 方法，支持是否套用 SMOTE 的選項
"""

def load_processed_data():
    df = pd.read_csv("mbti_full.csv", encoding='utf-8', encoding_errors='ignore')
    df['E/I'] = df['type'].str[0]
    df['S/N'] = df['type'].str[1]
    df['T/F'] = df['type'].str[2]
    df['J/P'] = df['type'].str[3]
    df = df.drop(columns=['type', 'posts', 'bert_vector', 'minilm_vector', 'type.1'], errors='ignore')
    assert 'type' not in df.columns, "'type' 欄位仍存在，請檢查是否正確移除"

    # MinMax Scaling for emotions
    scaler = MinMaxScaler()
    need_scaling = ['anger', 'anticipation', 'disgust','fear','joy','sadness','surprise','trust']
    df[need_scaling] = scaler.fit_transform(df[need_scaling])

    # One-hot encoding for VADER
    vader_one_hot = pd.get_dummies(df['vader_label'], prefix='vader')
    df = pd.concat([df.drop(columns=['vader_label']), vader_one_hot], axis=1)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # 特徵維度統計列印
    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    bert_cols = [col for col in df.columns if col.startswith("bert_") or col.startswith("minilm_")]
    sentiment_cols = ['vader_pos', 'vader_neu', 'vader_neg', 'vader_compound']
    label_cols = ['E/I', 'S/N', 'T/F', 'J/P']
    exclude_cols = tfidf_cols + bert_cols + sentiment_cols + label_cols + ['Unnamed: 0', 'type', 'posts']
    other_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    feature_set_1 = tfidf_cols + sentiment_cols + other_cols
    feature_set_2 = bert_cols + sentiment_cols + other_cols
    feature_set_3 = tfidf_cols + bert_cols + sentiment_cols + other_cols

    print("特徵維度統計：")
    print(f"1. TF-IDF + 詞語統計 + 情感分析：{len(feature_set_1)} 維")
    print(f"2. BERT + 詞語統計 + 情感分析：{len(feature_set_2)} 維")
    print(f"3. TF-IDF + BERT + 詞語統計 + 情感分析（全特徵）：{len(feature_set_3)} 維")
    return df

def get_feature_columns(df):
    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    bert_cols = [col for col in df.columns if col.startswith("bert_") or col.startswith("minilm_")]
    sentiment_cols = ['vader_pos', 'vader_neu', 'vader_neg', 'vader_compound']
    label_cols = ['E/I', 'S/N', 'T/F', 'J/P']
    exclude_cols = tfidf_cols + bert_cols + sentiment_cols + label_cols + ['Unnamed: 0', 'type', 'posts']
    other_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    return {
        'tfidf(set1)': tfidf_cols + sentiment_cols + other_cols,
        'bert(set2)': bert_cols + sentiment_cols + other_cols,
        'all(set3)': tfidf_cols + bert_cols + sentiment_cols + other_cols
    }

def split_data(df, dimension, feature_set, feature_dict, smote=False):
    X = df[feature_dict[feature_set]]
    y = df[dimension]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"資料切分設定：MBTI 維度 = {dimension}，特徵組合 = {feature_set}，SMOTE = {smote}")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    if smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test, le

