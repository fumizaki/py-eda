from enum import Enum

class LinRegTask(Enum):
    """Linear Regressionパイプラインのタスク種別"""
    REGRESSION = "回帰"
    # LinearRegressionは回帰専用なので、CLASSIFICATIONタスクは含めません

class LinRegMetric(Enum):
    """Linear Regressionパイプラインの評価指標"""
    # 回帰タスク用
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"