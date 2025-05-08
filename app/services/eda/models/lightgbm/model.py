import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Any, Optional, Union
import numpy as np
from .enum import TaskType, ClassificationMetric, MulticlassMetric, RegressionMetric, ClassificationObjective, MulticlassObjective, RegressionObjective
from .parameter import get_params


class LightGBMModel:
    """
    LightGBMモデルの学習、予測、評価を行うクラス。
    """
    def __init__(self):
        self.task_type = TaskType.CLASSIFICATION
        self.params: dict = get_params(self.task_type)

        # モデルインスタンス
        self.model: Union[lgb.LGBMClassifier, lgb.LGBMRegressor]
        if self.is_classification() or self.is_multiclass():
            self.model = lgb.LGBMClassifier(**self.params)
        elif self.is_regression():
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # データ保持用
        self.df: Optional[pd.DataFrame] = None
        self.feature_columns: list[str] = []
        self.target_column: Optional[str] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        self.eval_df: Optional[pd.DataFrame] = None

    def check_df_loaded(self):
        if self.df is None:
            raise ValueError("DataFrame is not loaded yet.")
        
    def task_options(self) -> list[str]:
        return [TaskType.CLASSIFICATION, TaskType.MULTICLASS, TaskType.REGRESSION]

    # --- タスクタイプ判定ユーティリティ ---
    def is_classification(self) -> bool:
        return self.task_type == TaskType.CLASSIFICATION

    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    def load(self, task_type: TaskType, df: pd.DataFrame, feature_columns: list[str], target_column: str, test_size: float = 0.2, random_state: int = 42) -> None:
        self.task_type = task_type
        self.params: dict = get_params(self.task_type)

        # モデルインスタンス
        self.model: Union[lgb.LGBMClassifier, lgb.LGBMRegressor]
        if self.is_classification() or self.is_multiclass():
            self.model = lgb.LGBMClassifier(**self.params)
        elif self.is_regression():
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        self.df = df
        self.target_column = target_column

        X = self.df[feature_columns]
        y = self.df[self.target_column]

        stratify = y if self.is_classification() or self.is_multiclass() else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        self.feature_columns = X.columns.tolist()

    def train(self, early_stopping_rounds: Optional[int] = None) -> None:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not loaded. Call load() first.")

        fit_params: dict[str, Any] = {}
        if early_stopping_rounds and self.X_test is not None and self.y_test is not None:
            fit_params['eval_set'] = [(self.X_test, self.y_test)]
            fit_params['callbacks'] = [lgb.early_stopping(early_stopping_rounds)]

        self.model.fit(self.X_train, self.y_train, **fit_params)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet. Please call train().")

        # 特徴量チェック & 整形
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Input data missing columns: {missing_cols}")

        X_processed = X[self.feature_columns]

        if self.is_classification():
            return self.model.predict(X_processed)
        elif self.is_multiclass():
            return self.model.predict_proba(X_processed)
        elif self.is_regression():
            return self.model.predict(X_processed)

    def evaluate(self) -> pd.DataFrame:
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not available.")

        eval_metrics: dict[str, float | np.ndarray] = {}

        if self.is_classification():
            y_pred = self.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test[self.feature_columns])[:, 1]

            eval_metrics = {
                ClassificationMetric.ACCURACY.value: accuracy_score(self.y_test, y_pred),
                ClassificationMetric.PRECISION.value: precision_score(self.y_test, y_pred, average='binary', zero_division=0),
                ClassificationMetric.RECALL.value: recall_score(self.y_test, y_pred, average='binary', zero_division=0),
                ClassificationMetric.F1.value: f1_score(self.y_test, y_pred, average='binary', zero_division=0),
                ClassificationMetric.AUC.value: roc_auc_score(self.y_test, y_pred_proba),
                ClassificationMetric.LOGLOSS.value: log_loss(self.y_test, y_pred_proba)
            }

        elif self.is_multiclass():
            # y_pred_proba = self.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test[self.feature_columns])
            y_pred = np.argmax(y_pred_proba, axis=1)

            eval_metrics = {
                MulticlassMetric.AUC.value: roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr'),
                MulticlassMetric.MULTI_LOGLOSS.value: log_loss(self.y_test, y_pred_proba)
            }

        elif self.is_regression():
            y_pred = self.predict(self.X_test)

            mse = mean_squared_error(self.y_test, y_pred)
            eval_metrics = {
                RegressionMetric.MSE.value: mse,
                RegressionMetric.RMSE.value: np.sqrt(mse),
                RegressionMetric.MAE.value: mean_absolute_error(self.y_test, y_pred),
                RegressionMetric.R2.value: r2_score(self.y_test, y_pred)
            }

        eval_df = pd.DataFrame.from_dict(eval_metrics, orient='index', columns=['Score'])

        self.eval_df = eval_df
        return eval_df

    def get_feature_importances(self) -> Optional[pd.DataFrame]:
        """
        モデルの特徴量重要度を DataFrame として取得する。

        Returns:
            Optional[pd.DataFrame]: 特徴量名 ('Feature' カラム) と重要度 ('Importance' カラム)
                                    を含む DataFrame。重要度の降順でソートされます。
                                    モデルが学習されていない場合や feature_importances_ がない場合は None を返す。
        """
        # モデルが存在し、かつ特徴量重要度属性があり、かつ特徴量のカラム名リストが存在する場合
        if self.model is not None and hasattr(self.model, 'feature_importances_') and self.feature_columns is not None:
            # 特徴量名と重要度の NumPy 配列から DataFrame を作成
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.model.feature_importances_
            })

            # 'Importance' カラムの値で降順にソート
            feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

            return feature_importance_df

        # 条件を満たさない場合は None を返す
        return None

    def cross_validate(self, cv: int = 5) -> dict[str, np.ndarray]:
        if self.df is None or self.target_column is None:
            raise ValueError("Data is not loaded. Call load().")

        X = self.df[self.feature_columns]
        y = self.df[self.target_column]

        if self.is_classification() or self.is_multiclass():
            model_cv = lgb.LGBMClassifier(**self.params)
        elif self.is_regression():
            model_cv = lgb.LGBMRegressor(**self.params)

        scoring = {}
        if self.is_classification():
            scoring = {
                ClassificationMetric.ACCURACY.value: 'accuracy',
                ClassificationMetric.AUC.value: 'roc_auc',
                ClassificationMetric.LOGLOSS.value: 'neg_log_loss'
            }
        elif self.is_multiclass():
            scoring = {
                MulticlassMetric.AUC.value: 'roc_auc_ovr',
                MulticlassMetric.MULTI_LOGLOSS.value: 'neg_log_loss'
            }
        elif self.is_regression():
            scoring = {
                RegressionMetric.MSE.value: 'neg_mean_squared_error',
                RegressionMetric.MAE.value: 'neg_mean_absolute_error',
                RegressionMetric.R2.value: 'r2'
            }

        cv_results: dict[str, np.ndarray] = {}
        for metric_name, scoring_method in scoring.items():
            scores = cross_val_score(model_cv, X, y, scoring=scoring_method, cv=cv, n_jobs=-1)
            if scoring_method.startswith('neg_'):
                scores = -scores
            cv_results[metric_name] = scores

            if metric_name == RegressionMetric.MSE.value:
                cv_results[RegressionMetric.RMSE.value] = np.sqrt(cv_results[RegressionMetric.MSE.value])

        return cv_results