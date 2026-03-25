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

def main():
    # 🌟 Sweepエージェントから呼び出されるたびにinitされ、ランダムに選ばれたconfigが渡されます
    wandb.init()
    config = wandb.config

    set_seed(42)

    # 1. データの読み込み
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../data/data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"⚠️ データファイルが見つかりません: {data_path}")

    df = pd.read_csv(data_path)

    # 特徴量の追加
    df["PerformanceToPayRatio"] = df["MonthlyAchievement"] / df["MonthlyIncome"].replace(0, np.nan)
    df["IncentiveRatio"] = df["Incentive"] / df["MonthlyIncome"].replace(0, np.nan)

    features = [
        'Age', 'MonthlyIncome', 'YearsAtCompany', 'YearsSinceLastPromotion', 
        'WorkLifeBalance', 'JobSatisfaction', 'StressRating',
        'PerformanceToPayRatio', 'IncentiveRatio'
    ]
    target = 'Attrition'

    X = df[features].copy()
    y = df[target]

    ordinal_mappings = {
        'WorkLifeBalance': {'Bad': 1, 'Good': 2, 'Better': 3, 'Best': 4},
        'JobSatisfaction': {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
        'StressRating': {
            "1. 'Very Low'": 1, "3. 'Average'": 3, "5. 'Very High'": 5
        }
    }

    for col, mapping in ordinal_mappings.items():
        if col in X.columns:
            X[col] = X[col].replace(mapping)

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    # 目的変数の変換
    if 'Yes' in y.values or 'No' in y.values:
        y = y.map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # 2. モデルの学習（今回は config の値を使用）
    model = lgb.LGBMClassifier(
        random_state=42,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        scale_pos_weight=config.scale_pos_weight,
        max_depth=config.max_depth,          # 木の深さ
        num_leaves=config.num_leaves,        # 葉の最大数
        verbose=-1 # 余計なログを消す
    )
    model.fit(X_train, y_train)

    # 3. 評価指標の計算と記録
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # logに渡した値の中から、sweep_configで指定した 'recall' を見て自動で評価されます！
    wandb.log({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })

if __name__ == "__main__":
    # === WandB Sweeps の設定 ===
    # どのようなアルゴリズムで、どのパラメータを探索するかを定義します
    sweep_config = {
        'method': 'bayes', # ベイズ最適化（効率よく最良な設定を探します。 'random' や 'grid' も可）
        'metric': {
            'name': 'recall',     # 最大化したい指標
            'goal': 'maximize'   
        },
        'parameters': {
            'learning_rate': {
                'min': 0.01,
                'max': 0.3
            },
            'scale_pos_weight': {
                'min': 1.0,
                'max': 10.0      # 離職クラスの重み(1倍～10倍まで探索)
            },
            'n_estimators': {
                'values': [50, 100, 200]
            },
            'max_depth': {
                'values': [3, 5, 7, -1] # -1 は無制限
            },
            'num_leaves': {
                'values': [15, 31, 63]
            }
        }
    }
    
    # 1. Sweepプロジェクトを初期化し、IDを発行（この設定をWandBサーバーに送ります）
    sweep_id = wandb.sweep(sweep_config, project="b4-training-GCI-lgbm-sweeps")
    
    # 2. 自動的に指定した回数（今回は count=20 回）だけパラメータを変えながら main 関数を実行します
    print("🚀 WandB Sweeps によるハイパーパラメータ探索を開始します！")
    wandb.agent(sweep_id, function=main, count=20)
