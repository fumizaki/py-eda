from .enum import MLPTask


def get_mlp_params(task: MLPTask) -> dict:
    static_params = {
        'hidden_layer_sizes': (100,), # デフォルトで中間層1つ、100ノード
        'activation': 'relu',        # 活性化関数: ReLU
        'solver': 'adam',            # 最適化アルゴリズム: Adam
        'alpha': 0.0001,             # L2ペナルティ（正則化項）
        'batch_size': 'auto',        # バッチサイズ
        'learning_rate': 'constant', # 学習率の調整方法
        'learning_rate_init': 0.001, # 初期学習率
        'max_iter': 200,             # エポック数（訓練データの反復回数）
        'random_state': 42,          # 乱数シード
        'tol': 1e-4,                 # 収束判定のための許容誤差
        'verbose': False,            # 学習プロセス表示
        'early_stopping': False,     # 検証セットに対する早期停止を使用するか
        'n_iter_no_change': 10,      # early_stopping=Trueの場合、スコア改善が見られないエポック数の閾値
        'max_fun': 15000,            # L-BFGSソルバーでの最大イテレーション回数（他のソルバーでは無視される）
    }

    specific_params = {
        MLPTask.BINARY: {},
        MLPTask.MULTICLASS: {},
        MLPTask.REGRESSION: {}
    }

    return {**static_params, **specific_params[task]}

