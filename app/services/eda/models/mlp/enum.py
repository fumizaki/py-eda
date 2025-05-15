from enum import Enum

class MLPTask(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class MLPObjective(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class MLPMetric(Enum):
    # Binary
    ACCURACY = "accuracy"
    AUC = "auc"
    LOGLOSS = "logloss"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    # Multiclass
    MULTI_LOGLOSS = "multi_logloss"
    MULTI_ERROR = "multi_error"
    # Regression
    RMSE = "rmse"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
