from .enum import ClassificationObjective, MulticlassObjective, RegressionObjective, ClassificationMetric, MulticlassMetric, RegressionMetric, TaskType


def get_classification_params() -> dict:
    return {
            'objective': ClassificationObjective.BINARY.value,
            'metric': ClassificationMetric.AUC.value,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }

def get_multiclass_params() -> dict:
    return {
            'objective': MulticlassObjective.MULTICLASS.value,
            'metric': MulticlassMetric.MULTI_LOGLOSS.value,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }

def get_regression_params() -> dict:
    return {
            'objective': RegressionObjective.RMSE.value,
            'metric': RegressionMetric.RMSE.value,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }

def get_params(task_type: TaskType) -> dict:

    if task_type == TaskType.CLASSIFICATION:
        return get_classification_params()
    
    elif task_type == TaskType.MULTICLASS:
        return get_multiclass_params()
    
    elif task_type == TaskType.REGRESSION:
        return get_regression_params()

    else:
        # 定義されていないタスクタイプの場合
        raise ValueError(f"Unsupported task type: {task_type}")

