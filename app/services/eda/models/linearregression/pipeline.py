import pandas as pd
import numpy as np
from typing import Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LinearRegression
from .enum import LinRegTask, LinRegMetric
from .parameter import get_linreg_params


class LinearRegressionPipeline:
    """
    線形回帰 (Linear Regression) モデルを使用した回帰パイプライン。
    データの読み込み、学習用/テスト用分割、モデル学習、予測、評価、係数、クロスバリデーションをサポート。
    """
    def __init__(self, task: Optional[LinRegTask] = LinRegTask.REGRESSION):
        """
        LinearRegressionPipelineを初期化します。

        Parameters
        ----------
        task : Optional[LinRegTask], default=LinRegTask.REGRESSION
            パイプラインのタスク種別 (REGRESSION)。
        """
        if task != LinRegTask.REGRESSION:
             raise ValueError("Linear Regression only supports REGRESSION task.")

        self.task: LinRegTask = task
        # random_stateはモデルパラメータではないが、パイプラインで保持
        self._random_state: int = 42
        self.params: dict = self.get_params()
        # モデルはタスクとパラメータに基づいて get_model() で初期化
        self.model: LinearRegression = self.get_model() # 線形回帰はRegressorのみ

        # データおよび分割されたデータを保持するための属性
        self.features: Optional[list[str]] = None  # 入力特徴量の名前のリスト
        self.target: Optional[str] = None  # 目的変数のカラム名
        self.df: Optional[pd.DataFrame] = None  # ロードされた元のDataFrame
        self.X_train: Optional[pd.DataFrame] = None  # 特徴量の学習データ
        self.X_test: Optional[pd.DataFrame] = None  # 特徴量のテストデータ
        self.y_train: Optional[pd.Series] = None  # 目的変数の学習データ
        self.y_test: Optional[pd.Series] = None  # 目的変数のテストデータ


    def load_task(self, task: LinRegTask) -> None:
        """
        パイプラインのタスク種別を変更し、パラメータとモデルを更新します。

        Parameters
        ----------
        task : LinRegTask
            新しいタスク種別。Linear RegressionではREGRESSIONのみ有効。

        Raises
        ------
        ValueError
            REGRESSION以外のタスクが指定された場合。
        """
        if task != LinRegTask.REGRESSION:
             raise ValueError("Linear Regression only supports REGRESSION task.")

        self.task = task
        self.params = self.get_params() # 新しいタスクのデフォルトパラメータを取得 (REGRESSION用)
        self.model = self.get_model()   # 新しいモデルを初期化


    def get_params(self) -> dict:
        """
        現在のタスク種別に基づいてデフォルトパラメータを取得します。

        Returns
        -------
        dict
            モデルのパラメータ辞書。
        """
        # Linear RegressionはREGRESSIONタスクのみを想定してパラメータを取得
        return get_linreg_params(LinRegTask.REGRESSION)


    def get_model(self) -> LinearRegression:
        """
        現在のタスク種別とパラメータに基づいてLinear Regressionモデルを初期化します。

        Returns
        -------
        LinearRegression
            初期化されたLinear Regressionモデルインスタンス。
        """
        # Linear Regressionは常にRegressor
        return LinearRegression(**self.params)


    def load_data(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str,
        test_size: Optional[float] = 0.2,
        random_state: Optional[int] = 42 # load_dataに渡されたrandom_stateをパイプラインで保持
    ) -> None:
        """
        回帰用のデータを学習用とテスト用データに分割します。

        Parameters
        ----------
        df : pd.DataFrame
            元となるデータセット。
        features : list[str]
            学習に使用する特徴量（カラム名）のリスト。
        target : str
            予測対象となる目的変数（ターゲット）のカラム名。
        test_size : Optional[float], default=0.2
            テストデータに割り当てる割合。
        random_state : Optional[int], default=42
            再現性を確保するための乱数シード。データ分割に使用されます。

        Raises
        ------
        ValueError
            指定されたターゲットカラムまたは特徴量カラムの一部がDataFrameに存在しない場合。
        TypeError
            ターゲットカラムが数値型（integerまたはfloat）でない場合（Linear Regressionは数値ターゲットを期待するため）。
        """
        # ターゲットカラムが存在するか確認
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not in DataFrame columns.")

        # 特徴量カラムが全て存在するか確認
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
             raise ValueError(f"Input DataFrame is missing required features: {list(missing_features)}")

        # 属性として保持
        self.df = df
        self.features = features
        self.target = target
        self._random_state = random_state # パイプラインのrandom_stateを更新

        # 特徴量と目的変数を抽出
        X = df[features]
        y = df[target]

        # 線形回帰は数値ターゲット（int/float）を期待します
        if not pd.api.types.is_numeric_dtype(y):
             raise TypeError(
                 f"Target column '{target}' is of type {y.dtype} (not numeric). "
                 f"Linear Regression requires numerical targets. Please ensure the target column is numeric before calling load_data()."
             )

        # 回帰タスクでは層化抽出は通常行いません
        stratify = None

        # 学習データとテストデータに分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self._random_state, stratify=stratify
        )


    def fit(self) -> None:
        """
        学習データ (self.X_train, self.y_train) を用いてLinear Regressionモデルを訓練します。
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call load_data() first.")

        # LinearRegressionモデルの fit メソッドを呼び出す
        self.model.fit(self.X_train, self.y_train)


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        学習済みのLinear Regressionモデルを用いて、新しいデータに対する予測を行います。

        Parameters
        ----------
        X : pd.DataFrame
            予測に使用する特徴量データ。
            モデルを学習した際に指定した self.features と同じカラム構成である必要があります。

        Returns
        -------
        np.ndarray
            予測結果（回帰値）。 predict() の結果を返します。

        Raises
        ------
        ValueError
            入力データに、モデルが必要とする特徴量（self.features）の一部が欠けている場合。
        """
        if self.features is None:
             raise ValueError("Model has not been trained yet. Feature list is not available.")

        missing = set(self.features) - set(X.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {list(missing)}")

        X_processed = X[self.features]

        # 線形回帰は predict_proba は持っていません
        return self.model.predict(X_processed)


    def evaluate(self) -> pd.DataFrame:
        """
        テストデータ (self.X_test, self.y_test) を用いてモデルの性能を評価します。

        Returns
        -------
        pd.DataFrame
            評価指標とそのスコアを含むDataFrame。

        Raises
        ------
        ValueError
            テストデータが利用できない場合（load_data() が未実行またはテストサイズが0）。
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not available. Call load_data() with test_size > 0.")

        # 予測回帰値を取得
        y_pred = self.predict(self.X_test)

        metrics = {} # 評価結果を格納する辞書

        # Linear Regressionは回帰専用
        if self.task == LinRegTask.REGRESSION:
            mse = mean_squared_error(self.y_test, y_pred)
            metrics = {
                LinRegMetric.MSE.value: mse,
                LinRegMetric.RMSE.value: np.sqrt(mse), # MSEからRMSEを計算
                LinRegMetric.MAE.value: mean_absolute_error(self.y_test, y_pred),
                LinRegMetric.R2.value: r2_score(self.y_test, y_pred)
            }

        else:
             # このパイプラインは回帰専用なので、ここに到達することはないはず
             raise ValueError(f"Unsupported task during evaluation: {self.task}")


        return pd.DataFrame.from_dict(metrics, orient="index", columns=["Score"])


    def feature_coefficients(self) -> pd.DataFrame:
        """
        学習済みモデルから各特徴量の係数を取得します。
        線形回帰の係数は特徴量の予測への寄与度（正負含む）として解釈できます。

        Returns
        -------
        pd.DataFrame
            特徴量名と係数を含むDataFrame。

        Raises
        ------
        ValueError
            モデルが訓練されていない、または係数情報が取得できない場合。
        """
        if self.model is None or not hasattr(self.model, "coef_") or not hasattr(self.model, "intercept_"):
            raise ValueError("モデルが訓練されていないか、係数情報が利用できません。")
        if self.features is None:
             raise ValueError("特徴量リストが利用できません。load_data()が実行されていませんか？")


        # Linear Regressionの係数 (coef_) と切片 (intercept_)
        # coef_ は通常 (n_features,) の形状になります
        # intercept_ はスカラーまたは (1,) の形状になります
        # LinearRegressionでは coef_ は1次元配列になるはずです。

        coefficients = self.model.coef_ # 係数を取得

        # 切片 (Intercept) は特徴量重要度とは少し意味合いが違うため、DataFrameには含めないか別途表示
        # intercept = self.model.intercept_


        # 特徴量名と係数をDataFrameにまとめる
        # coef_ が (n_features,) の1次元配列であることを想定
        if coefficients.ndim > 1:
             # 想定外の形状の場合（例えば、Multitargetの場合など）
             # ここでは単一ターゲットの線形回帰を想定します。
             raise NotImplementedError(f"Coefficients with shape {coefficients.shape} display is not yet supported.")

        coefficients_df = pd.DataFrame({
            "feature": self.features,
            "coefficient": coefficients
        }).sort_values(by="coefficient", ascending=False) # 係数の値で降順にソート (正負あり)

        return coefficients_df.reset_index(drop=True)


    def cross_validate(
        self,
        cv: Optional[int] = 5
    ) -> pd.DataFrame:
        """
        K分割交差検証を実行してモデルの性能を評価します。

        Parameters
        ----------
        cv : Optional[int], default=5
            クロスバリデーションの分割数。

        Returns
        -------
        pd.DataFrame
            各分割での評価スコアを含むDataFrame。

        Raises
        ------
        ValueError
            データがロードされていない場合（load_data() が未実行）。
        """
        # データが設定されているかチェック（load_data()の呼び出しが必要）
        if self.df is None or self.target is None or self.features is None:
            raise ValueError("Data is not prepared. Call load_data() first.")

        # 説明変数（X）と目的変数（y）を取得
        X = self.df[self.features]
        y = self.df[self.target]

        # 新たにモデルをインスタンス化（毎回CVごとに同一条件で評価するため）
        # self.get_model() を呼び出すことで、self.params が反映されたモデルが作成されます。
        model = self.get_model()

        # scikit-learnのscoring文字列を定義
        # 線形回帰は回帰専用
        scoring_dict: dict[str, str] = {
            LinRegMetric.MSE.value: 'neg_mean_squared_error', # MSEは負の値で返る
            LinRegMetric.MAE.value: 'neg_mean_absolute_error', # MAEは負の値で返る
            LinRegMetric.R2.value: 'r2'
        }

        results: dict[str, Any] = {} # 各評価指標のCVスコアを格納する辞書

        # 指定された全ての評価指標でcross_val_scoreを実行
        for metric_name, scoring_method in scoring_dict.items():
            try:
                # cross_val_score を実行
                scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv = cv,
                    scoring = scoring_method,
                    n_jobs = -1, # 利用可能な全てのCPUコアを使用
                    # error_score = 'raise' # エラーが発生した場合に例外を投げる
                )

                # 'neg_' で始まる scoring はスコアが負の値で返るため、正に変換
                if scoring_method.startswith("neg_"):
                    scores = -scores

                # 結果を格納
                results[metric_name] = scores

                # MSE から RMSE を手動で計算して追加（回帰タスクのみ）
                if metric_name == LinRegMetric.MSE.value:
                    results[LinRegMetric.RMSE.value] = np.sqrt(scores)


            except Exception as e:
                print(f"Warning: Could not compute cross-validation score for metric '{metric_name}' using scoring '{scoring_method}'. Error: {str(e)}")
                # エラーが発生したメトリックの結果はNaNなどで埋める
                results[metric_name] = np.full(cv, np.nan) # エラー時はNaNの配列で埋める


        # 結果をDataFrame形式で返す（各列が評価指標、各行がCVの分割結果）
        return pd.DataFrame(results)