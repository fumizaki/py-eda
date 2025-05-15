from enum import Enum

class LogiRegTask(Enum):
    """Logistic Regressionパイプラインのタスク種別"""
    CLASSIFICATION = "分類"

class LogiRegMetric(Enum):
    """Logistic Regressionパイプラインの評価指標"""
    # 分類タスク用
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC = "auc" # 二値/多クラス両対応 (計算方法選択要)
    LOGLOSS = "logloss" # 二値/多クラス両対応