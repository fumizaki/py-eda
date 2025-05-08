from enum import Enum

class TaskType(Enum):
    """機械学習タスクのタイプ"""
    CLASSIFICATION = "Classification"
    MULTICLASS = "Multiclass"
    REGRESSION = "Regression"

    @classmethod
    def keys(cls) -> list[str]:
        return [i.name for i in cls]

    @classmethod
    def values(cls) -> list[str]:
        return [i.value for i in cls]


class ClassificationObjective(Enum):
    """LightGBM分類タスクの目的関数"""
    BINARY = "binary"

class MulticlassObjective(Enum):
    """LightGBM 多クラス分類の目的関数"""
    MULTICLASS = "multiclass"

class RegressionObjective(Enum):
    """LightGBM回帰タスクの目的関数"""
    REGRESSION = "regression"
    REGRESSION_L1 = "regression_l1"
    HUBER = "huber"
    FAIR = "fair"
    POISSON = "poisson"
    QUANTILE = "quantile"
    MAEL = "mael"
    RMSE = "rmse"

class ClassificationMetric(Enum):
    """二値分類タスクの評価指標"""
    ACCURACY = "accuracy"
    AUC = "auc"
    LOGLOSS = "logloss"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"

class MulticlassMetric(Enum):
    """多クラス分類タスクの評価指標"""
    MULTI_LOGLOSS = "multi_logloss"
    MULTI_ERROR = "multi_error"
    ACCURACY = "accuracy"
    AUC = "auc"

class RegressionMetric(Enum):
    """回帰タスクの評価指標"""
    RMSE = "rmse"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"

class PreprocessingStep(Enum):
    """前処理ステップのタイプ"""
    DROP_COLUMNS = "Drop Columns"
    FILL_NA_MEAN = "Fill NA (Mean)"
    FILL_NA_MEDIAN = "Fill NA (Median)"
    FILL_NA_MODE = "Fill NA (Mode)"
    FILL_NA_CONSTANT = "Fill NA (Constant)"
    LABEL_ENCODING = "Label Encoding"
    ONE_HOT_ENCODING = "One-Hot Encoding"
    MIN_MAX_SCALING = "Min-Max Scaling"
    STANDARD_SCALING = "Standard Scaling"

class FeatureEngineeringStep(Enum):
    """特徴量エンジニアリングステップのタイプ"""
    DATETIME_FEATURES = "Datetime Features"
    POLYNOMIAL_FEATURES = "Polynomial Features"
    INTERACTION_FEATURES = "Interaction Features"
    GROUP_BY_FEATURES = "Group By Features"