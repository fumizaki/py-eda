from enum import Enum

class RFTask(Enum):
    """Random Forestパイプラインのタスク種別"""
    CLASSIFICATION = "分類"
    REGRESSION = "回帰"

class RFMetric(Enum):
    """Random Forestパイプラインの評価指標"""
    # 分類タスク用
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC = "auc"
    LOGLOSS = "logloss" # 二値分類用
    MULTI_LOGLOSS = "multi_logloss" # 多クラス分類用

    # 回帰タスク用
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"
