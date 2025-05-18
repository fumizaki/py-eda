from typing import Optional
from .enum import LGBMTask, LGBMMetric, LGBMObjective


def get_lgbm_params(task: LGBMTask, num_class: Optional[int] = None) -> dict:
    static_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

    if task == LGBMTask.MULTICLASS:
        if num_class is None:
            raise ValueError("MULTICLASS タスクでは num_class を指定してください")

    specific_params = {
        LGBMTask.BINARY: {
            'objective': LGBMObjective.BINARY.value,
            'metric': LGBMMetric.AUC.value
        },
        LGBMTask.MULTICLASS: {
            'objective': LGBMObjective.MULTICLASS.value,
            'metric': LGBMMetric.MULTI_LOGLOSS.value,
            'num_class': num_class
        },
        LGBMTask.REGRESSION: {
            'objective': LGBMObjective.REGRESSION.value,
            'metric': LGBMMetric.RMSE.value
        }
    }

    return {**static_params, **specific_params[task]}

