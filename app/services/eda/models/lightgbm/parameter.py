from .enum import LGBMTask, LGBMMetric, LGBMObjective


def get_lgbm_params(task: LGBMTask) -> dict:
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

    specific_params = {
        LGBMTask.BINARY: {
            'objective': LGBMObjective.BINARY.value,
            'metric': LGBMMetric.AUC.value
        },
        LGBMTask.MULTICLASS: {
            'objective': LGBMObjective.MULTICLASS.value,
            'metric': LGBMMetric.MULTI_LOGLOSS.value
        },
        LGBMTask.REGRESSION: {
            'objective': LGBMObjective.REGRESSION.value,
            'metric': LGBMMetric.RMSE.value
        }
    }

    return {**static_params, **specific_params[task]}

