import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

class SentimentPreprocessor:
    def __init__(self, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        print(f"使用裝置：{self.device}")
        
        # BERT 模型初始化
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device).eval()
        
        # MiniLM 模型初始化
        self.minilm_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.minilm_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"MiniLM 模型是否使用 GPU: {self.minilm_model.device}")
        
        # Vader 分析器初始化
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def get_bert_cls_vector(self, text, max_length=512):
        inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        return cls_vector.squeeze().cpu().numpy()

    def get_minilm_mean_vector(self, text, chunk_token_limit=512):
        tokens = self.minilm_tokenizer.tokenize(text)
        chunks = [tokens[i:i + chunk_token_limit] for i in range(0, len(tokens), chunk_token_limit)]
        chunk_texts = [self.minilm_tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]
        embeddings = self.minilm_model.encode(chunk_texts, convert_to_tensor=True, device=self.device)
        mean_vector = torch.mean(embeddings, dim=0)
        return mean_vector.cpu().numpy()

    def analyze_sentiment_vader(self, text):
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        label = 'positive' if compound >= 0.05 else 'negative' if compound <= -0.05 else 'neutral'
        return pd.Series([scores['pos'], scores['neu'], scores['neg'], compound, label],
                         index=['vader_pos', 'vader_neu', 'vader_neg', 'vader_compound', 'vader_label'])

    def transform_dataframe(self, df, text_column='posts'):
        tqdm.pandas()
        print(f"SentimentPreprocessor: 開始向量化與情感分析（BERT + MiniLM + Vader）")

        # BERT CLS 向量
        df['bert_vector'] = df[text_column].progress_apply(self.get_bert_cls_vector)

        # MiniLM 平均向量
        df['minilm_vector'] = df[text_column].progress_apply(self.get_minilm_mean_vector)

        # Vader 情緒分析
        vader_results = df[text_column].progress_apply(self.analyze_sentiment_vader)
        df = pd.concat([df, vader_results], axis=1)

        print("SentimentPreprocessor: 處理完成")
        return df


    def save_npz(self, df, filename='sentiment_vectors.npz', label_column='type'):
        print("SentimentPreprocessor: 儲存向量中...")
        np.savez_compressed(
            filename,
            bert_vectors=np.stack(df['bert_vector'].values),
            minilm_vectors=np.stack(df['minilm_vector'].values),
            labels=df[label_column].values,
            vader_pos=df['vader_pos'].values.astype(np.float32),
            vader_neu=df['vader_neu'].values.astype(np.float32),
            vader_neg=df['vader_neg'].values.astype(np.float32),
            vader_compound=df['vader_compound'].values.astype(np.float32),
            vader_label=df['vader_label'].values.astype(str)
        )
        print(f"SentimentPreprocessor: 儲存至 {filename}")

