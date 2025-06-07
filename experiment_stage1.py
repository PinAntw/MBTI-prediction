# experiment_stage1.py
from data_utils import load_processed_data, get_feature_columns, split_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def run_feature_comparison():
    df = load_processed_data()
    feature_dict = get_feature_columns(df)

    feature_sets = ['tfidf', 'bert', 'all']
    dimensions = ['E/I', 'S/N', 'T/F', 'J/P']
    models = {
        'lr': LogisticRegression(max_iter=1000),
        'rf': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    }

    best_scores = {}

    for fs in feature_sets:
        print(f"\n===== Feature Set: {fs.upper()} =====")
        scores = []
        for dim in dimensions:
            print(f"-- Dimension: {dim} --")
            X_train, X_test, y_train, y_test = split_data(df, dim, fs, feature_dict, smote=False)

            for name, model in models.items():
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                print(f"[{name}]\n" + classification_report(y_test, y_pred))

    print("\n完成 Stage 1 特徵組合比較。建議根據結果選定最佳 feature_set 作為後續實驗基準。")
    return  # 可回傳最佳 feature_set 字串，例如 'all'
