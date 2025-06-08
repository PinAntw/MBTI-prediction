from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
import numpy as np
import nltk

# 自定義模組
from linguisticPreprocessor import LinguisticPreprocessor
from sentimentPreprocessor import SentimentPreprocessor
from tfidfPreprocessor import TfidfPreprocessor

# NLTK 資料路徑設定
NLTK_DATA_PATH = os.path.abspath("nltk_data")
nltk.data.path.append(NLTK_DATA_PATH)

app = FastAPI()

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型與向量器載入
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
model_IE = joblib.load(os.path.join(MODEL_DIR, "xgb_EI.pkl"))
model_SN = joblib.load(os.path.join(MODEL_DIR, "xgb_SN.pkl"))
model_TF = joblib.load(os.path.join(MODEL_DIR, "xgb_TF.pkl"))
model_JP = joblib.load(os.path.join(MODEL_DIR, "xgb_JP.pkl"))
le_IE = joblib.load(os.path.join(MODEL_DIR, "label_encoder_EI.pkl"))
le_SN = joblib.load(os.path.join(MODEL_DIR, "label_encoder_SN.pkl"))
le_TF = joblib.load(os.path.join(MODEL_DIR, "label_encoder_TF.pkl"))
le_JP = joblib.load(os.path.join(MODEL_DIR, "label_encoder_JP.pkl"))
tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

# 特徵欄位順序
with open(os.path.join(MODEL_DIR, "feature_columns.txt"), "r") as f:
    expected_columns = [line.strip() for line in f.readlines()]

# POST 請求資料格式
class PostInput(BaseModel):
    post: str

@app.post("/api/predict")
def predict(data: PostInput):
    input_text = data.post
    print(f"接收到文本：{input_text}")

    # === 防呆處理 ===
    if not input_text or input_text.strip() == "":
        raise HTTPException(status_code=400, detail="post 內容不可為空")
    if not isinstance(input_text, str):
        raise HTTPException(status_code=400, detail="post 必須為字串")
    input_text = input_text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")

    # === 前處理：情感分析特徵 ===
    sp = SentimentPreprocessor()
    df_sent = sp.transform_dataframe(pd.DataFrame([{'posts': input_text}]), text_column="posts")

    # === TF-IDF 特徵 ===
    tfidf_vec = tfidf_vectorizer.transform([input_text])
    tfidf_df = pd.DataFrame(tfidf_vec.toarray(), columns=[f'tfidf_{w}' for w in tfidf_vectorizer.get_feature_names_out()])

    # === 合併所有特徵 ===
    feature_row = pd.concat([
        df_sent.drop(columns=['type', 'posts', 'type.1'], errors='ignore'),
        tfidf_df
    ], axis=1).fillna(0)

    # === 對齊欄位順序 ===
    feature_row = feature_row.reindex(columns=expected_columns, fill_value=0)

    # === 修正 dtype ===
    for col in feature_row.columns:
        if feature_row[col].dtype == "object":
            feature_row[col] = feature_row[col].astype(str)
        if feature_row[col].dtype not in [np.int64, np.float64, np.bool_]:
            feature_row[col] = pd.to_numeric(feature_row[col], errors='coerce').fillna(0)

    # === 預測四個維度，還原為字母 ===
    result = {
        "IE": le_IE.inverse_transform(model_IE.predict(feature_row))[0],
        "SN": le_SN.inverse_transform(model_SN.predict(feature_row))[0],
        "TF": le_TF.inverse_transform(model_TF.predict(feature_row))[0],
        "JP": le_JP.inverse_transform(model_JP.predict(feature_row))[0],
    }

    print("預測結果：", result)
    return result
