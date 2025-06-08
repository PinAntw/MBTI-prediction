# MBTI 性格預測系統

本專案旨在透過使用者的文字內容（如社群貼文）預測其 MBTI（Myers-Briggs Type Indicator）人格特質。我們將此任務拆解為四個二元分類問題，並透過多種特徵組合與機器學習模型進行比較與整合。

你可以在下列網址直接體驗我們的預測系統：
[MBTI預測系統](https://mbti-prediction.vercel.app/)
## 專案背景

MBTI 模型將人格分為四個向度，共十六種人格類型：

- E / I：外向（Extroversion）與內向（Introversion）
- S / N：實感（Sensing）與直覺（Intuition）
- T / F：思考（Thinking）與情感（Feeling）
- J / P：判斷（Judging）與知覺（Perceiving）

本系統針對每一個向度建立獨立分類器，以支援個別預測與整體組合推斷。

## 使用特徵

本系統整合以下多種特徵組合：

- TF-IDF 詞頻特徵
- NRC 字典情緒統計
- VADER 極性情感分數
- BERT / MiniLM 語意向量嵌入（Sentence Embedding）

## 實驗設計

實驗流程包含以下三個階段：

1. **特徵組合比較**：比較不同語意與情緒特徵對模型效能之影響  
2. **SMOTE 類別不平衡處理**：觀察是否對部分人格向度使用 SMOTE 能改善效能  
3. **模型融合**：結合 Logistic Regression 與 KNN 模型進行 soft voting，評估是否優於單一模型

## 使用模型

- Logistic Regression
- SVM（使用高斯核函數）
- Random Forest
- XGBoost
- Soft Voting Ensemble

## 執行方式

請依照下列步驟執行本地端實驗流程：

### 1. 複製專案
git clone https://github.com/PinAntw/MBTI-prediction.git
cd MBTI-prediction

### 2. 安裝所需 Python 套件
pip install -r requirements.txt

### 3. 執行主程式進行實驗
python main.py

## 專案結構說明

```
mbti_prediction/
├── main.py                   # 主實驗控制腳本（特徵組合、模型比較、結果輸出）
├── model/                    # 預訓練模型與向量器儲存區
│   ├── model_IE.pkl
│   ├── model_SN.pkl
│   ├── model_TF.pkl
│   └── model_JP.pkl
├── sentimentPreprocessor.py  # 情感極性特徵前處理
├── linguisticPreprocessor.py # 情緒詞彙統計特徵處理（NRC 等）
├── tfidfPreprocessor.py      # TF-IDF 特徵處理器
├── data_utils.py             # 資料載入與切分函數
├── requirements.txt          # Python 執行所需套件清單
└── mbti_full.csv             # 預設輸入資料集（MBTI 人格與貼文）
```

## 參考文獻

- Zhang, H. (2023). MBTI personality prediction based on BERT classification. *Highlights in Science, Engineering and Technology*, 34, 138–143.

- Zumma, M. T., Munia, J. A., Halder, D., & Rahman, M. S. (2022, October). Personality prediction from Twitter dataset using machine learning. In *2022 13th International Conference on Computing, Communication and Networking Technologies (ICCCNT)* (pp. 1–7). IEEE.

- Zhou, Y., Shi, J., & Yu, Q. (2022). MBTI personality analysis and prediction. *Big Data Analysis Final Report*, Columbia University.