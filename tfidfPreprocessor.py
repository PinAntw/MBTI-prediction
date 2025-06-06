import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfPreprocessor:
    def __init__(self, max_features=1000, ngram_range=(1, 2), stop_words='english'):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=self.stop_words,
            ngram_range=self.ngram_range
        )

    def clean_text(self, text):
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()

    def preprocess_posts(self, texts):
        posts = [p.strip() for p in texts.split('|||') if p.strip()]
        combined_text = " ".join(posts)
        return self.clean_text(combined_text)

    def generate_features(self, df, text_column='posts', label_column='type'):
        print("TfidfPreprocessor: 正在產生 TF-IDF 特徵")

        # 預處理貼文
        processed_texts = df[text_column].apply(self.preprocess_posts)

        # TF-IDF 向量化
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        feature_names = self.vectorizer.get_feature_names_out()

        # 建立特徵 DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{name}' for name in feature_names]
        )
        tfidf_df[label_column] = df[label_column].values

        print(f"TfidfPreprocessor: 特徵維度 {tfidf_df.shape}")
        return tfidf_df

    def save_features(self, tfidf_df, filename='mbti_tfidf_features.csv', drop_columns=None):
        if drop_columns:
            tfidf_df = tfidf_df.drop(columns=drop_columns, errors='ignore')
        tfidf_df.to_csv(filename, index=False)
        print(f"TfidfPreprocessor: 特徵已儲存至 {filename}")
