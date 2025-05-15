from .enum import LinRegTask


def get_linreg_params(task: LinRegTask) -> dict:
    """
    Linear Regressionモデルのデフォルトパラメータをタスクに基づいて返します。

    Parameters
    ----------
    task : LinRegTask
        Linear Regressionのタスク種別 (REGRESSION)。

    Returns
    -------
    dict
        Linear Regressionモデルのデフォルトパラメータ。

    Raises
    ------
    ValueError
        REGRESSION以外のタスクが指定された場合。
    """
    # Linear Regressionは比較的シンプルなパラメータを持ちます。
    static_params = {
        'fit_intercept': True,      # 切片を学習するかどうか
        'copy_X': True,             # Xをコピーするかどうか
        'n_jobs': -1,               # 訓練に使用するCPUコア数 (-1は全て)
        # random_stateはLinearRegressionのパラメータにはありませんが、データ分割で使用するため考慮
        # 'random_state': 42, # パイプライン側で管理
    }

    # Linear RegressionはREGRESSIONタスクのみを想定
    specific_params = {
        LinRegTask.REGRESSION: {
            # 回帰タスク固有のパラメータはほとんどない
        }
    }

    if task == LinRegTask.REGRESSION:
        # LinearRegressionはrandom_stateをパラメータとして受け取らないため、
        # パイプライン側で保持し、load_dataなどで使用します。
        # ここではget_paramsとしては返さず、static_paramsのみとします。
        return {**static_params, **specific_params[task]}
    else:
        # Linear Regressionは回帰以外のタスクはサポートしない
        raise ValueError(f"Unsupported task for Linear Regression: {task}")
