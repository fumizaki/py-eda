from enum import Enum

class TaskType(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class Objective(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class Metric(Enum):
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