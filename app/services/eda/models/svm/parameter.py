from .enum import SVMTask


def get_svm_params(task: SVMTask) -> dict:
    """
    SVMモデル (SVC/SVR) のデフォルトパラメータをタスクに基づいて返します。

    Parameters
    ----------
    task : SVMTask
        SVMのタスク種別 (CLASSIFICATION, REGRESSION)。

    Returns
    -------
    dict
        SVMモデルのデフォルトパラメータ。

    Raises
    ------
    ValueError
        サポートされていないタスクが指定された場合。
    """
    # SVMに共通するパラメータ
    static_params = {
        'C': 1.0,                   # 正則化パラメータ
        'kernel': 'rbf',            # カーネルの種類 ('linear', 'poly', 'rbf', 'sigmoid')
        'gamma': 'scale',           # カーネル係数 ('scale' or 'auto')
        'shrinking': True,          # シュリンキングヒューリスティックを使用するか
        'tol': 1e-3,                # 収束判定の許容誤差
        'max_iter': -1,             # ソルバーの最大反復回数 (-1は制限なし)
        'random_state': 42,         # 乱数シード (確率推定や一部のソルバーで使用)
    }

    # タスク固有のパラメータ
    specific_params = {
        SVMTask.CLASSIFICATION: {
            'probability': True,    # 確率推定を有効にするか (AUC, LogLossに必要。有効にすると遅くなる)
            'class_weight': None,   # クラスの重み (不均衡データに有効)
            # 'decision_function_shape': 'ovr', # 多クラス分類の方法 ('ovr' or 'ovo') - 'ovr'がデフォルト
        },
        SVMTask.REGRESSION: {
            'epsilon': 0.1,         # ε-SVRモデルのパラメータ
        }
    }

    if task in [SVMTask.CLASSIFICATION, SVMTask.REGRESSION]:
        # タスク固有のパラメータを共通パラメータとマージ
        return {**static_params, **specific_params[task]}
    else:
        raise ValueError(f"Unsupported task for SVM params: {task}")