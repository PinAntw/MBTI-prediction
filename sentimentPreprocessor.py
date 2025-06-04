import os
import re
import statistics
from collections import defaultdict, Counter

import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize

class SentimentPreprocessor:
    def __init__(self, nltk_data_path="./nltk_data"):
        self.nltk_data_path = nltk_data_path
        self.emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        self.word_to_emotions = None

    def setup_nltk(self):
        nltk_paths = [
            os.path.join(self.nltk_data_path, "tokenizers", "punkt", "english.pickle"),
            os.path.join(self.nltk_data_path, "tokenizers", "punkt_tab"),
            os.path.join(self.nltk_data_path, "taggers", "averaged_perceptron_tagger", "averaged_perceptron_tagger.pickle"),
            os.path.join(self.nltk_data_path, "taggers", "averaged_perceptron_tagger_eng"),
        ]

        if all(os.path.exists(p) for p in nltk_paths):
            print("NLTK 資源已完整，略過下載嘍")
            nltk.data.path.append(self.nltk_data_path)
            return

        os.makedirs(self.nltk_data_path, exist_ok=True)
        nltk.data.path.append(self.nltk_data_path)

        nltk.download("punkt", download_dir=self.nltk_data_path)
        nltk.download("punkt_tab", download_dir=self.nltk_data_path)
        nltk.download("averaged_perceptron_tagger", download_dir=self.nltk_data_path)
        nltk.download("averaged_perceptron_tagger_eng", download_dir=self.nltk_data_path)


    def load_nrc_lexicon(self, folder_path):
        emotion_dict = {emotion: set() for emotion in self.emotions}
        for emotion in self.emotions:
            file_path = os.path.join(folder_path, f"{emotion}-NRC-Emotion-Lexicon.txt")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        word, flag = parts
                        if flag == '1':
                            emotion_dict[emotion].add(word.lower())
        self.word_to_emotions = emotion_dict

    def remove_links(self, text):
        return re.sub(r'https?://\S+', '', text)

    def count_links(self, text):
        return len(re.findall(r'https?://\S+', text))

    def safe_std(self, lst):
        return statistics.stdev(lst) if len(lst) > 1 else 0

    def ratio_or_zero(self, count, total_tagged_words):
        return count / total_tagged_words if total_tagged_words > 0 else 0

    def clean_text(self, p):
        p = self.remove_links(p.strip())
        return re.sub(r'\s+', ' ', p)

    def preprocess(self, texts):
        posts = [p.strip() for p in texts.split('|||') if p.strip() != '']
        combined_text = " ".join(posts)
        post_clean = re.sub(r'\s+', ' ', combined_text).strip()
        post_no_links = self.remove_links(post_clean)
        num_posts = len(posts)
        return posts, post_clean, post_no_links, num_posts

    def process_posts(self, texts):
        posts, post_clean, post_no_links, num_posts = self.preprocess(texts)

        post_lengths = [len(self.clean_text(post).split()) for post in posts]
        std_words = self.safe_std(post_lengths)

        emoticon_pattern = r'[:;=8][-^]?[)D(\]/\\OpP]|XD|xD|orz|qq|QQ|:-?\)|:-?D|:-?P|:-?O|:-?/|:\^\)|:\$|:\'\(|:\*\)|:P|:\)|:\(|:D|;\)|;\(|:\||:\^\)|=\)|=\(|=D|=P|=O|;D|:O|:\]|;-\)|:3|:\>|\(:|\|@@\||@\.@|@_@|@-@|@o@'

        tag_categories = {
            'pronouns': {'PRP', 'PRP$'},
            'nouns': {'NN', 'NNS', 'NNP', 'NNPS'},
            'verbs': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
            'adjectives': {'JJ', 'JJR', 'JJS'},
            'interjections': {'UH'},
            'prep_conj': {'IN', 'CC'},
            'wh_words': {'WP', 'WP$'},
            'adverbs': {'RB', 'RBR', 'RBS'}
        }

        words = word_tokenize(post_no_links)
        tagged = pos_tag(words)
        tag_counts = Counter(tag for _, tag in tagged)

        counts = {cat + '_count': sum(tag_counts[tag] for tag in tags) for cat, tags in tag_categories.items()}
        counts['total_tagged_words'] = sum(tag_counts.values())

        words_all_post = len(words)
        question_marks = post_no_links.count('?')
        exclamation_marks = post_no_links.count('!')
        ellipsis_counts = post_no_links.count('...')
        tilde_counts = post_no_links.count('~')
        dash_counts = post_no_links.count('-') + post_no_links.count('—')
        links_counts = self.count_links(post_clean)
        char_counts = len(post_no_links)
        emoticon_counts = len(re.findall(emoticon_pattern, post_no_links))
        sentence_counts = len(sent_tokenize(post_no_links))

        all_words = [w.lower() for w in words if w.isalpha()]
        vocab_richness = len(set(all_words)) / len(all_words) if all_words else 0

        feature_dict = {
            'std_words': std_words,
            'vocab_richness': vocab_richness,
            'avg_words': words_all_post / num_posts if num_posts else 0,
            'avg_sentences': sentence_counts / num_posts if sentence_counts else 0,
            'avg_question_marks': question_marks / num_posts if num_posts else 0,
            'avg_exclamation_marks': exclamation_marks / num_posts if num_posts else 0,
            'avg_ellipsis': ellipsis_counts / num_posts if num_posts else 0,
            'avg_tilde': tilde_counts / num_posts if num_posts else 0,
            'avg_dash': dash_counts / num_posts if num_posts else 0,
            'avg_links': links_counts / num_posts if links_counts else 0,
            'avg_chars': char_counts / num_posts if char_counts else 0,
            'avg_emojis': emoticon_counts / num_posts if emoticon_counts else 0,
        }

        for cat in tag_categories:
            feature_dict[cat + '_ratio'] = self.ratio_or_zero(counts[cat + '_count'], words_all_post)

        return pd.Series(feature_dict)

    def LIWC_NRC_detail(self, text):
        posts, post_clean, post_no_links, num_posts = self.preprocess(text)
        result = defaultdict(int)
        text_words = post_no_links.lower().split()
        for word in text_words:
            for emotion in self.word_to_emotions.get(word, []):
                result[emotion] += 1
        return pd.Series({emotion: result.get(emotion, 0) for emotion in self.emotions})

    def generate_features(self, df, text_column):
        print("SentimentPreprocessor: 開始資料集文本情感前處理")
        text_series = df[text_column]
        post_features = text_series.apply(self.process_posts)
        emotion_features = text_series.apply(self.LIWC_NRC_detail)
        print("SentimentPreprocessor: 執行完畢")
        return pd.concat([df.reset_index(drop=True), post_features, emotion_features], axis=1)

    def process_single_text(self, text):
        print("SentimentPreprocessor: 開始處理新文本")
        post_feat = self.process_posts(text)
        emotion_feat = self.LIWC_NRC_detail(text)
        print("SentimentPreprocessor: 處理新文本完畢")
        return pd.concat([post_feat, emotion_feat])