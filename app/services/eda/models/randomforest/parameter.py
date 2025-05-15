from .enum import RFTask, RFMetric


def get_rf_params(task: RFTask) -> dict:
    static_params = {
        'n_estimators': 100,         # 決定木の数
        'max_depth': None,           # 木の最大深度 (Noneは制限なし)
        'min_samples_split': 2,      # 分割に必要な最小サンプル数
        'min_samples_leaf': 1,       # 葉に必要な最小サンプル数
        'max_features': 'sqrt',      # 各分割で考慮する特徴量の数
        'random_state': 42,          # 乱数シード
        'n_jobs': -1,                # 訓練に使用するCPUコア数 (-1は全て)
        'ccp_alpha': 0.0             # 最小コスト複雑度剪定のパラメータ
    }

    specific_params = {
        RFTask.CLASSIFICATION: {
            'criterion': 'gini', # 分類基準 (情報利得)
            'oob_score':  True   # OOBサンプルでスコアを計算
        },
        RFTask.REGRESSION: {
            'criterion': 'squared_error' # 回帰基準 (MSE)
        }
    }

    return {**static_params, **specific_params[task]}

