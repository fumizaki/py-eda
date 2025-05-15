import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Optional, Union, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, mean_squared_error,
    mean_absolute_error, r2_score
)
from .enum import LGBMTask, LGBMMetric
from .parameter import get_lgbm_params


class LGBMPipeline:
    def __init__(self, task: Optional[LGBMTask] = LGBMTask.BINARY):
        self.task: LGBMTask = task  # タスクの種類
        self.params: dict = self.get_params()
        self.model: Union[lgb.LGBMClassifier, lgb.LGBMRegressor] = self.get_model()
        self.features: Optional[list[str]] = []  # 入力特徴量の名前
        self.target: Optional[str] = None  # 目的変数のカラム名
        self.df: Optional[pd.DataFrame] = None  # 元のDataFrame
        self.X_train: Optional[pd.DataFrame] = None  # 特徴量の学習データ
        self.X_test: Optional[pd.DataFrame] = None  # 特徴量のテストデータ
        self.y_train: Optional[pd.Series] = None  # 目的変数の学習データ
        self.y_test: Optional[pd.Series] = None  # 目的変数のテストデータ

    # タスクの種類
    def load_task(self, task: LGBMTask) -> None:
        self.task = task
        self.params = self.get_params()
        self.model = self.get_model()


    # パラメータ取得
    def get_params(self) -> dict:
        return get_lgbm_params(self.task)


    # モデルの初期化（分類 or 回帰）
    def get_model(self) -> Union[lgb.LGBMClassifier, lgb.LGBMRegressor]:
        if self.task in [LGBMTask.BINARY, LGBMTask.MULTICLASS]:
            return lgb.LGBMClassifier(**self.params)
        elif self.task == LGBMTask.REGRESSION:
            return lgb.LGBMRegressor(**self.params)
        else:
            raise ValueError(f"Unsupported task: {self.task}")


    # データを学習用とテスト用に分割するメソッド
    def load_data(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str,
        test_size: Optional[float] = 0.2,
        random_state: Optional[float] = 42
    ) -> None:
        """
        指定された特徴量と目的変数をもとに、学習用データとテスト用データに分割します。

        Parameters:
        ----------
        df : pd.DataFrame
            元となるデータセット。
        features : list[str]
            学習に使用する特徴量（カラム名）のリスト。
        target : str
            予測対象となる目的変数（ターゲット）のカラム名。
        test_size : Optional[float], default=0.2
            テストデータに割り当てる割合（例：0.2なら全体の20%がテスト用）。
        random_state : Optional[float], default=42
            再現性を確保するための乱数シード。

        Raises:
        ------
        ValueError:
            指定されたターゲットカラムが DataFrame に存在しない場合。
        TypeError:
            分類タスクなのにターゲットカラムが文字列型（object型）の場合。

        Returns:
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            学習用特徴量、テスト用特徴量、学習用目的変数、テスト用目的変数の順のタプル。
        """

        # 目的変数が存在するか確認
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not in DataFrame columns.")
        
        # 属性として保持
        self.df = df
        self.features = features
        self.target = target

        # 特徴量と目的変数を抽出
        X = df[features]
        y = df[target]

        # 分類タスクかつ目的変数が object 型（例：文字列ラベル）の場合はエラーを出す
        if self.task in [LGBMTask.BINARY, LGBMTask.MULTICLASS] and y.dtype == "object":
            raise TypeError(
                f"Target column '{target}' is of type object (e.g., strings like 'setosa'). "
                f"Please encode the labels as integers (e.g., using LabelEncoder) before calling prepare_data()."
            )

        # 分類タスクでは層化抽出を行う
        stratify = y if self.task in [LGBMTask.BINARY, LGBMTask.MULTICLASS] else None

        # 学習データとテストデータに分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )



    def fit(self, early_stopping_rounds: Optional[int] = None) -> None:
        """
        学習データを用いて LightGBM モデルを訓練するメソッド。

        Parameters
        ----------
        early_stopping_rounds : int, optional
            アーリーストッピング（Early Stopping）を有効にするためのラウンド数。
            指定した整数値が与えられた場合、検証用データ（X_test, y_test）を使って、
            指定されたラウンド数の間、評価指標が改善しなければ訓練を自動で停止する。
            これにより過学習（overfitting）を防ぎ、学習時間を短縮できる。

        Raises
        ------
        ValueError
            学習データが用意されていない場合（load_data() が未実行）、エラーを発生させる。
        """

        # 学習用データが未設定の場合はエラーを出す
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call load_data() first.")

        fit_args = {}  # fit メソッドに渡す追加引数をまとめる辞書

        # アーリーストッピングが設定されていて、検証用データが存在する場合
        if early_stopping_rounds and self.X_test is not None:
            fit_args["eval_set"] = [(self.X_test, self.y_test)]  # 評価用データを設定
            fit_args["callbacks"] = [lgb.early_stopping(early_stopping_rounds)]  # アーリーストッピング用のコールバックを指定

        # LightGBMモデルを訓練データで学習させる
        self.model.fit(self.X_train, self.y_train, **fit_args)

    # 予測関数
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        学習済みの LightGBM モデルを用いて、新しいデータに対する予測を行うメソッド。

        Parameters
        ----------
        X : pd.DataFrame
            予測に使用する特徴量データ。
            モデルを学習した際に指定した self.features と同じカラム構成である必要がある。

        Returns
        -------
        np.ndarray
            予測結果。
            - タスクが MULTICLASS の場合は、各クラスに対する確率を含む 2次元配列（predict_proba）。
            - それ以外のタスク（BINARY または REGRESSION）の場合は、予測値の 1次元配列（predict）。

        Raises
        ------
        ValueError
            入力データに、モデルが必要とする特徴量（self.features）の一部が欠けている場合。
        """

        # モデルが学習時に使用した特徴量と、予測時に与えられた特徴量の整合性を確認する
        missing = set(self.features) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")  # 欠損しているカラム名を明示してエラーを出す

        # 学習時と同じ順序・構成で特徴量を抽出
        X_processed = X[self.features]

        # タスクが MULTICLASS（多クラス分類）の場合、各クラスに対する確率を返す
        if self.task == LGBMTask.MULTICLASS:
            return self.model.predict_proba(X_processed)  # 各クラスの確率を返す（行×クラス数の配列）

        # BINARY または REGRESSION の場合は、直接予測値を返す
        return self.model.predict(X_processed)  # 1次元の予測値配列


    # モデルの性能を評価
    def evaluate(self) -> pd.DataFrame:
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not available.")
        y_pred = self.predict(self.X_test)
        metrics = {}

        # ===== BINARY（2値分類） =====
        if self.task == LGBMTask.BINARY:
            proba = self.model.predict_proba(self.X_test)[:, 1]  # 正例の確率
            metrics = {
                # ==== 分類タスク: BINARY ====
                "accuracy": accuracy_score(self.y_test, y_pred), 
                # 正解率：全予測のうち正解だった割合

                "precision": precision_score(self.y_test, y_pred), 
                # 適合率：正と予測した中で実際に正だった割合（偽陽性を防ぎたい時に重視）

                "recall": recall_score(self.y_test, y_pred), 
                # 再現率：実際に正であったもののうち正と予測できた割合（偽陰性を防ぎたい時に重視）

                "f1": f1_score(self.y_test, y_pred), 
                # F1スコア：適合率と再現率の調和平均。バランス良く評価したいときに使う

                "auc": roc_auc_score(self.y_test, proba), 
                # ROC AUCスコア：真陽性率と偽陽性率のトレードオフの曲線下の面積。1に近いほど良い

                "logloss": log_loss(self.y_test, proba)
                # ロジスティック損失：予測確率と実際の差を対数で評価（小さいほど良い）
            }

        # ===== MULTICLASS（多クラス分類） =====
        elif self.task == LGBMTask.MULTICLASS:
            proba = self.model.predict_proba(self.X_test)

            metrics = {
                "multi_logloss": log_loss(self.y_test, proba)  # 通常はこれでOK
            }

            try:
                metrics["auc"] = roc_auc_score(self.y_test, proba, multi_class='ovr')
            except ValueError as e:
                metrics["auc"] = np.nan

        # ===== REGRESSION（回帰） =====
        elif self.task == LGBMTask.REGRESSION:
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
            "feature": self.features,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        if top_k is not None:
            importance_df = importance_df.head(top_k)  # 上位k件を抽出

        return importance_df.reset_index(drop=True)
    

    def cross_validate(
        self,
        cv: Optional[int] = 5
    ) -> pd.DataFrame:
        # データが設定されているかチェック（prepare_data()の呼び出しが必要）
        if self.df is None or self.target is None:
            raise ValueError("Data is not prepared.")

        # 説明変数（X）と目的変数（y）を取得
        X = self.df[self.features]
        y = self.df[self.target]

        # 新たにモデルをインスタンス化（毎回CVごとに同一条件で評価）
        model = self.get_model()

        scoring_dict: dict[str, str] = {}
        # タスクごとに適切な評価指標（scoring）を指定
        if self.task == LGBMTask.BINARY:
            scoring_dict = {
                LGBMMetric.ACCURACY.value: 'accuracy',  # 正解率
                LGBMMetric.AUC.value: 'roc_auc',  # AUC（2値分類の性能）
                LGBMMetric.LOGLOSS.value: 'neg_log_loss'  # ロジスティック損失（マイナスで返るため後で反転）
            }
        elif self.task == LGBMTask.MULTICLASS:
            scoring_dict = {
                LGBMMetric.AUC.value: 'roc_auc_ovr',  # マルチクラス用のAUC（One-vs-Rest）
                LGBMMetric.MULTI_LOGLOSS.value: 'neg_log_loss'  # ロジスティック損失
            }
        elif self.task == LGBMTask.REGRESSION:
            scoring_dict = {
                LGBMMetric.MSE.value: 'neg_mean_squared_error',  # MSE（二乗誤差）
                LGBMMetric.MAE.value: 'neg_mean_absolute_error',  # MAE（絶対誤差）
                LGBMMetric.R2.value: 'r2'  # R2スコア（決定係数）
            }

        results: dict[str, Any] = {}

        # 指定された全ての評価指標でcross_val_scoreを実行
        for metric, method in scoring_dict.items():
            try:
                scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv = cv,
                    scoring = method,
                    n_jobs = -1,
                    # error_score = 'raise'
                )
            except Exception as e:
                print(f"Error for scoring={method}: {str(e)}")
                raise ValueError(e)

            # neg_xxx 形式は符号が逆なので正に変換
            if method.startswith("neg_"):
                scores = -scores

            # 結果を格納
            results[metric] = scores

            # MSE から RMSE を手動で計算して追加（回帰タスクのみ）
            if self.task == LGBMTask.REGRESSION and metric == LGBMMetric.MSE.value:
                results[LGBMMetric.RMSE.value] = np.sqrt(scores)

        # 結果をDataFrame形式で返す（評価指標×スコア配列）
        return pd.DataFrame(results)