from .enum import TaskType, Objective, Metric


def get_default_params(task_type: TaskType) -> dict:
    common = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

    task_specific = {
        TaskType.BINARY: {
            'objective': Objective.BINARY.value,
            'metric': Metric.AUC.value
        },
        TaskType.MULTICLASS: {
            'objective': Objective.MULTICLASS.value,
            'metric': Metric.MULTI_LOGLOSS.value
        },
        TaskType.REGRESSION: {
            'objective': Objective.REGRESSION.value,
            'metric': Metric.RMSE.value
        }
    }

    return {**common, **task_specific[task_type]}

