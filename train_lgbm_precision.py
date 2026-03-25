import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
import lightgbm as lgb
import wandb

def set_seed(seed):
    """すべての乱数シードを固定する関数"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- ⚙️ 実験用設定 ---
DRY_RUN = False
RESUME_TRAINING = False

# 1. wandbの初期化（実験の記録開始）
# プロジェクト名をLGBM用に変更
wandb.init(project="b4-training-GCI-lgbm", config={
    "model_type": "LightGBM",
    "random_state": 42,
    "n_estimators": 100 if not DRY_RUN else 10,
    "test_size": 0.1,
    "learning_rate": 0.11444508979757552,
    "scale_pos_weight": 9.807512236718487,
    "num_leaves": 63,
    "max_depth": 3,
})
config = wandb.config

# 乱数シードの固定
set_seed(config.random_state)

# 2. データの準備
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "../data/data.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"⚠️ データファイルが見つかりません: {data_path}")

df = pd.read_csv(data_path)

if DRY_RUN:
    df = df.sample(n=min(100, len(df)), random_state=config.random_state)
    print("🛑 [DRY RUN] データを100件に絞って実行します。")

# === 新規特徴量の追加 ===
df["PerformanceToPayRatio"] = df["MonthlyAchievement"] / df["MonthlyIncome"].replace(0, np.nan)
df["IncentiveRatio"] = df["Incentive"] / df["MonthlyIncome"].replace(0, np.nan)

# 3. 特徴量と目的変数の定義
features = [
    'Age', 
    'MonthlyIncome', 
    'YearsAtCompany', 
    'YearsSinceLastPromotion', 
    'WorkLifeBalance', 
    'JobSatisfaction', 
    'StressRating',
    'PerformanceToPayRatio',
    'IncentiveRatio'
]
target = 'Attrition'

X = df[features].copy()
y = df[target]

# Column.md に定義された順序付きカテゴリの数値化
ordinal_mappings = {
    'WorkLifeBalance': {'Bad': 1, 'Good': 2, 'Better': 3, 'Best': 4},
    'JobSatisfaction': {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
    'StressRating': {
        "1. 'Very Low'": 1, "3. 'Average'": 3, "5. 'Very High'": 5, 
        "1. 'Very Low'": 1, "3. 'Average'": 3, "5. 'Very High'": 5
    }
}

for col, mapping in ordinal_mappings.items():
    if col in X.columns:
        X[col] = X[col].replace(mapping)

# 万が一の他の文字列用エラー回避
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(0)

# 目的変数の変換：一律の置換ルール（計算を伴わないためリークにはなりません）
if 'Yes' in y.values or 'No' in y.values:
    y = y.map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
elif y.dtype == 'object' or y.dtype.name == 'category':
    # 文字列の場合は全体にfitさせないよう、カテゴリ型に変換だけしておき、LightGBMに任せます
    y = y.astype('category').cat.codes

# 4. データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=config.test_size, 
    random_state=config.random_state, 
    stratify=y
)

# LightGBMは木の分岐を用いるため、基本的に事前の標準化（StandardScaler）は不要です。

# 5. モデルの構築と学習
model = lgb.LGBMClassifier(
    random_state=config.random_state,
    n_estimators=config.n_estimators,      # ← これがエポック数(木の数)に相当します
    learning_rate=config.learning_rate,    # ← 追加: 学習率
    scale_pos_weight=config.scale_pos_weight, # ← 追加: 離職クラスの重みを直接指定
    num_leaves=config.num_leaves,          # ← 追加: 葉の最大数
    max_depth=config.max_depth             # ← 追加: 木の深さ
)
model.fit(X_train, y_train)

# 6. 予測
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 7. 評価指標の算出と表示
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("=== 評価指標 (LightGBM) ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# wandbに評価指標を記録
wandb.log({
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc
})

print("🎉 学習プロセス完了！")
wandb.finish()
