import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, mean_squared_error,
    mean_absolute_error, r2_score
)
from .enum import TaskType, Metric
from .parameter import get_default_params


class LGBMPipeline:
    def __init__(self, task: TaskType = TaskType.BINARY):
        self.task = task  # タスクの種類
        self.params = get_default_params(task)  # パラメータ取得
        self.model = self._init_model()  # モデルの初期化（分類 or 回帰）
        self.feature_names = []  # 入力特徴量の名前
        self.target_name = None  # 目的変数のカラム名
        self.df = None  # 元のDataFrame
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # タスクに応じたモデル（分類 or 回帰）を初期化
    def _init_model(self):
        if self.task in [TaskType.BINARY, TaskType.MULTICLASS]:
            return lgb.LGBMClassifier(**self.params)
        elif self.task == TaskType.REGRESSION:
            return lgb.LGBMRegressor(**self.params)
        raise ValueError(f"Unsupported task: {self.task}")

    # データを学習用とテスト用に分割
    def prepare_data(self, task: TaskType, df: pd.DataFrame, features: list[str], target: str, test_size: Optional[float] = 0.2, random_state: Optional[float] = 42):
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not in DataFrame columns.")
        self.df = df
        self.feature_names = features
        self.target_name = target
        
        self.task = task
        self.params = get_default_params(self.task)
        self.model = self._init_model()
        
        X = df[features]
        y = df[target]
        # 目的変数が文字列型のままならエラーを出す
        if self.task in [TaskType.BINARY, TaskType.MULTICLASS] and y.dtype == "object":
            raise TypeError(
                f"Target column '{target}' is of type object (e.g., strings like 'setosa'). "
                f"Please encode the labels as integers (e.g., using LabelEncoder) before calling prepare_data()."
            )
        stratify = y if self.task in [TaskType.BINARY, TaskType.MULTICLASS] else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

    # モデル学習（early stopping も可能）
    def fit(self, early_stopping_rounds: Optional[int] = None):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        fit_args = {}
        if early_stopping_rounds and self.X_test is not None:
            fit_args["eval_set"] = [(self.X_test, self.y_test)]
            fit_args["callbacks"] = [lgb.early_stopping(early_stopping_rounds)]
        self.model.fit(self.X_train, self.y_train, **fit_args)

    # 予測関数
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        X_processed = X[self.feature_names]
        if self.task == TaskType.MULTICLASS:
            return self.model.predict_proba(X_processed)  # 多クラスでは確率を返す
        return self.model.predict(X_processed)  # それ以外は通常の予測

    # モデルの性能を評価
    def evaluate(self) -> pd.DataFrame:
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not available.")
        y_pred = self.predict(self.X_test)
        metrics = {}

        # ===== BINARY（2値分類） =====
        if self.task == TaskType.BINARY:
            proba = self.model.predict_proba(self.X_test)[:, 1]  # 正例の確率
            metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),  # 正解率
                "precision": precision_score(self.y_test, y_pred),  # 適合率：正と予測したうち正解の割合
                "recall": recall_score(self.y_test, y_pred),  # 再現率：実際に正のうち予測できた割合
                "f1": f1_score(self.y_test, y_pred),  # F1スコア：precisionとrecallのバランス
                "auc": roc_auc_score(self.y_test, proba),  # AUCスコア：ROC曲線の下の面積
                "logloss": log_loss(self.y_test, proba)  # ロジスティック損失：予測確率の誤差
            }

        # ===== MULTICLASS（多クラス分類） =====
        elif self.task == TaskType.MULTICLASS:
            proba = self.model.predict_proba(self.X_test)

            metrics = {
                "multi_logloss": log_loss(self.y_test, proba)  # 通常はこれでOK
            }

            try:
                metrics["auc"] = roc_auc_score(self.y_test, proba, multi_class='ovr')
            except ValueError as e:
                metrics["auc"] = np.nan

        # ===== REGRESSION（回帰） =====
        elif self.task == TaskType.REGRESSION:
            mse = mean_squared_error(self.y_test, y_pred)
            metrics = {
                "mse": mse,  # 平均二乗誤差：誤差の2乗の平均
                "rmse": np.sqrt(mse),  # RMSE：誤差の平均的な大きさ
                "mae": mean_absolute_error(self.y_test, y_pred),  # 平均絶対誤差
                "r2": r2_score(self.y_test, y_pred)  # 決定係数：1に近いほど説明力が高い
            }

        return pd.DataFrame.from_dict(metrics, orient="index", columns=["Score"])  # 結果をDataFrameに整形

    # 特徴量の重要度を取得
    def feature_importances(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        top_k: 上位k件を取得（Noneの場合は全て）
        """
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            raise ValueError("モデルが訓練されていない、または重要度情報が取得できません。")

        importances = self.model.feature_importances_  # LightGBMによる特徴量の重要度
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        if top_k is not None:
            importance_df = importance_df.head(top_k)  # 上位k件を抽出

        return importance_df.reset_index(drop=True)

    def cross_validate(self, cv: int = 5) -> pd.DataFrame:
        # データが設定されているかチェック（prepare_data()の呼び出しが必要）
        if self.df is None or self.target_name is None:
            raise ValueError("Data is not prepared.")

        # 説明変数（X）と目的変数（y）を取得
        X = self.df[self.feature_names]
        y = self.df[self.target_name]

        # 新たにモデルをインスタンス化（毎回CVごとに同一条件で評価）
        model = self._init_model()

        # タスクごとに適切な評価指標（scoring）を指定
        if self.task == TaskType.BINARY:
            scoring = {
                Metric.ACCURACY.value: 'accuracy',  # 正解率
                Metric.AUC.value: 'roc_auc',  # AUC（2値分類の性能）
                Metric.LOGLOSS.value: 'neg_log_loss'  # ロジスティック損失（マイナスで返るため後で反転）
            }
        elif self.task == TaskType.MULTICLASS:
            scoring = {
                Metric.AUC.value: 'roc_auc_ovr',  # マルチクラス用のAUC（One-vs-Rest）
                Metric.MULTI_LOGLOSS.value: 'neg_log_loss'  # ロジスティック損失
            }
        elif self.task == TaskType.REGRESSION:
            scoring = {
                Metric.MSE.value: 'neg_mean_squared_error',  # MSE（二乗誤差）
                Metric.MAE.value: 'neg_mean_absolute_error',  # MAE（絶対誤差）
                Metric.R2.value: 'r2'  # R2スコア（決定係数）
            }

        results = {}

        # 指定された全ての評価指標でcross_val_scoreを実行
        for metric, method in scoring.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=method, n_jobs=-1)

            # neg_xxx 形式は符号が逆なので正に変換
            if method.startswith("neg_"):
                scores = -scores

            # 結果を格納
            results[metric] = scores

            # MSE から RMSE を手動で計算して追加
            if metric == Metric.MSE.value:
                results[Metric.RMSE.value] = np.sqrt(scores)

        # 各評価指標ごとの分割スコアを含むDataFrameを返す
        return pd.DataFrame(results)
