import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
import lightgbm as lgb
import wandb

def set_seed(seed):
    """すべての乱数シードを固定する関数"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# --- ⚙️ 実験用設定 ---
# ✅ ドライラン： 本番流しの前に、エラーが出ないか少量のデータ(100件程度)だけで回して強制終了するテストモード
DRY_RUN = False
# ✅ レジューム： 学習済みの重みから再開するか（今回はベースラインのロジスティック回帰なので基本的に使いません）
RESUME_TRAINING = False

# 1. wandbの初期化（実験の記録開始）
wandb.init(project="b4-training-GCI-basemodel", config={
    "model_type": "LogisticRegression",
    "random_state": 42,
    "max_iter": 1000 if not DRY_RUN else 10,
    "test_size": 0.1,
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
    # ドライランの時は少量のデータのみサンプリングして実行
    df = df.sample(n=min(100, len(df)), random_state=config.random_state)
    print("🛑 [DRY RUN] データを100件に絞って実行します。")

# === 新規特徴量の追加 ===
df["PerformanceToPayRatio"] = df["MonthlyAchievement"] / df["MonthlyIncome"].replace(0, np.nan)

# ※ Incentiveがもし数値でなく文字列であった場合は、先に数値化する必要がありますが、いったんそのまま計算します
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

# Column.md に定義された順序付きカテゴリの数値化（元の特徴量のみに適用）
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

# 万が一、モデル投入時に他の文字列が残っている場合のエラー回避
X = pd.get_dummies(X, drop_first=True)

# 割り算などによって発生した欠損値(NaN)が存在するとエラーになるため、0で埋めておく
X = X.fillna(0)

# 目的変数の変換：文字列 'Yes'/'No' または数値として存在している場合に対応
if 'Yes' in y.values or 'No' in y.values:
    y = y.map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
elif y.dtype == 'object' or y.dtype.name == 'category':
    from sklearn.preprocessing import LabelEncoder
    y = pd.Series(LabelEncoder().fit_transform(y))

# 4. データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=config.test_size, 
    random_state=config.random_state, 
    stratify=y
)

# [追加] ロジスティック回帰の収束改善のためにデータを標準化します
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. モデルの構築と学習
model = LogisticRegression(
    random_state=config.random_state, 
    max_iter=config.max_iter
)
model.fit(X_train, y_train)

# --- 参考: LightGBM を使用する場合 ---
# model_lgb = lgb.LGBMClassifier(random_state=config.random_state)
# model_lgb.fit(X_train, y_train)
# -----------------------------------

# 6. 予測
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 7. 評価指標の算出と表示
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("=== 評価指標 (Logistic Regression) ===")
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
