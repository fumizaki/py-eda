from .enum import LogiRegTask


def get_logireg_params(task: LogiRegTask) -> dict:
    """
    Logistic Regressionモデルのデフォルトパラメータをタスクに基づいて返します。

    Parameters
    ----------
    task : LRTask
        Logistic Regressionのタスク種別 (CLASSIFICATION)。

    Returns
    -------
    dict
        Logistic Regressionモデルのデフォルトパラメータ。
    """
    # Logistic Regressionは分類専用なので、タスク別のパラメータはほぼありません。
    # static_paramsのみを使用する形式に近いですが、将来的な拡張性のため形式を保ちます。
    static_params = {
        'penalty': 'l2',            # 正則化の種類 ('l1', 'l2', 'elasticnet', 'none')
        'C': 1.0,                   # 正則化の強さ (小さいほど強い正則化)
        'solver': 'lbfgs',          # 最適化アルゴリズム
        'max_iter': 100,            # ソルバーの最大反復回数
        'random_state': 42,         # 乱数シード
        'n_jobs': -1,               # 訓練に使用するCPUコア数 (-1は全て)
        'class_weight': None,       # クラスの重み ('balanced'など)
        'multi_class': 'auto',      # 多クラス分類の扱い ('auto'で自動判別)
        'verbose': 0                # 冗長性レベル
    }

    # Logistic RegressionはCLASSIFICATIONタスクのみを想定
    specific_params = {
        LogiRegTask.CLASSIFICATION: {
            # 現状、Logistic Regressionではタスク固有で切り替える必須パラメータは少ない
            # 例: multi_classはタスクによって自動判断させることが多い
        }
    }

    if task == LogiRegTask.CLASSIFICATION:
        return {**static_params, **specific_params[task]}
    else:
        # Logistic Regressionは分類以外のタスクはサポートしない
        raise ValueError(f"Unsupported task for Logistic Regression: {task}")


