from enum import Enum

class SVMTask(Enum):
    """SVMパイプラインのタスク種別"""
    CLASSIFICATION = "分類"
    REGRESSION = "回帰"

class SVMMetric(Enum):
    """SVMパイプラインの評価指標"""
    # 分類タスク用
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC = "auc" # 二値/多クラス両対応 (計算方法選択要)
    LOGLOSS = "logloss" # 二値/多クラス両対応

    # 回帰タスク用
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"