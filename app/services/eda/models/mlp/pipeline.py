import pandas as pd
import numpy as np
from typing import Optional, Union, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, mean_squared_error,
    mean_absolute_error, r2_score
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from .enum import MLPTask, MLPMetric
from .parameter import get_mlp_params



class MLPPipeline:
    """
    多層パーセプトロン (MLP) モデルを使用した機械学習パイプライン。
    データの読み込み、学習用/テスト用分割、モデル学習、予測、評価、クロスバリデーションをサポート。
    """
    def __init__(self, task: Optional[MLPTask] = MLPTask.BINARY):
        """
        MLPPipelineを初期化します。

        Parameters
        ----------
        task : Optional[MLPTask], default=MLPTask.BINARY
            パイプラインのタスク種別 (二値分類、多クラス分類、回帰)。
        """
        # タスクの種類を設定し、それに基づいてパラメータとモデルを初期化
        self.task: MLPTask = task
        self.params: dict = self.get_params()
        self.model: Union[MLPClassifier, MLPRegressor] = self.get_model()

        # データおよび分割されたデータを保持するための属性
        self.features: Optional[list[str]] = None  # 入力特徴量の名前のリスト
        self.target: Optional[str] = None  # 目的変数のカラム名
        self.df: Optional[pd.DataFrame] = None  # ロードされた元のDataFrame
        self.X_train: Optional[pd.DataFrame] = None  # 特徴量の学習データ
        self.X_test: Optional[pd.DataFrame] = None  # 特徴量のテストデータ
        self.y_train: Optional[pd.Series] = None  # 目的変数の学習データ
        self.y_test: Optional[pd.Series] = None  # 目的変数のテストデータ


    def load_task(self, task: MLPTask) -> None:
        """
        パイプラインのタスク種別を変更し、パラメータとモデルを更新します。

        Parameters
        ----------
        task : MLPTask
            新しいタスク種別。
        """
        self.task = task
        self.params = self.get_params()
        self.model = self.get_model()


    def get_params(self) -> dict:
        """
        現在のタスク種別に基づいてデフォルトパラメータを取得します。

        Returns
        -------
        dict
            モデルのパラメータ辞書。
        """
        # MLPのパラメータを取得
        return get_mlp_params(self.task)


    def get_model(self) -> Union[MLPClassifier, MLPRegressor]:
        """
        現在のタスク種別とパラメータに基づいてMLPモデルを初期化します。

        Returns
        -------
        Union[MLPClassifier, MLPRegressor]
            初期化されたMLPモデルインスタンス。

        Raises
        ------
        ValueError
            サポートされていないタスク種別が設定されている場合。
        """
        if self.task in [MLPTask.BINARY, MLPTask.MULTICLASS]:
            # 分類モデルを初期化
            return MLPClassifier(**self.params)
        elif self.task == MLPTask.REGRESSION:
            # 回帰モデルを初期化
            return MLPRegressor(**self.params)
        else:
            # サポートされていないタスク種別に対するエラー
            raise ValueError(f"Unsupported task for MLP model: {self.task}")


    def load_data(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str,
        test_size: Optional[float] = 0.2,
        random_state: Optional[int] = 42 # random_stateはint推奨
    ) -> None:
        """
        指定された特徴量と目的変数をもとに、DataFrameを学習用とテスト用データに分割します。

        Parameters
        ----------
        df : pd.DataFrame
            元となるデータセット。
        features : list[str]
            学習に使用する特徴量（カラム名）のリスト。
        target : str
            予測対象となる目的変数（ターゲット）のカラム名。
        test_size : Optional[float], default=0.2
            テストデータに割り当てる割合（例：0.2なら全体の20%がテスト用）。
        random_state : Optional[int], default=42
            再現性を確保するための乱数シード。
        
        Raises
        ------
        ValueError
            指定されたターゲットカラムまたは特徴量カラムの一部がDataFrameに存在しない場合。
        TypeError
            分類タスクなのにターゲットカラムが数値型（integerまたはfloat）でない場合。
        """
        # ターゲットカラムが存在するか確認
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not in DataFrame columns.")
            
        # 特徴量カラムが全て存在するか確認
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
             raise ValueError(f"Missing feature columns: {missing_features}")

        # 属性として保持
        self.df = df
        self.features = features
        self.target = target

        # 特徴量と目的変数を抽出
        X = df[features]
        y = df[target]

        # 分類タスクかつ目的変数が数値型でない場合のエラーチェック
        # MLPClassifierはint/floatラベルを期待するため
        if self.task in [MLPTask.BINARY, MLPTask.MULTICLASS] and not pd.api.types.is_numeric_dtype(y):
             raise TypeError(
                 f"Target column '{target}' is of type {y.dtype} (not numeric). "
                 f"For classification tasks with MLP, please encode the labels as integers (e.g., using LabelEncoder) before calling load_data()."
             )


        # 分類タスクでは層化抽出を行う（データ分割の際にクラス比率を維持）
        stratify = y if self.task in [MLPTask.BINARY, MLPTask.MULTICLASS] else None

        # 学習データとテストデータに分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )


    def fit(self) -> None:
        """
        学習データ (self.X_train, self.y_train) を用いてMLPモデルを訓練します。
        """
        # 学習用データが未設定の場合はエラーを出す
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call load_data() first.")

        # MLPClassifier/MLPRegressorの fit メソッドを呼び出す
        # scikit-learnのMLPはパラメータ（max_iter, early_stopping, n_iter_no_changeなど）で
        # 内部的に学習制御を行うため、MLPのような callbacks 引数は不要です。
        self.model.fit(self.X_train, self.y_train)


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        学習済みのMLPモデルを用いて、新しいデータに対する予測を行います。

        Parameters
        ----------
        X : pd.DataFrame
            予測に使用する特徴量データ。
            モデルを学習した際に指定した self.features と同じカラム構成である必要があります。

        Returns
        -------
        np.ndarray
            予測結果。
            - タスクが MULTICLASS または BINARY の場合は、各クラスに対する確率を含む 2次元配列（predict_proba）。
            - タスクが REGRESSION の場合は、予測値の 1次元配列（predict）。

        Raises
        ------
        ValueError
            入力データに、モデルが必要とする特徴量（self.features）の一部が欠けている場合。
        """
        # モデルが学習時に使用した特徴量と、予測時に与えられた特徴量の整合性を確認する
        if self.features is None:
             raise ValueError("Model has not been trained yet. Feature list is not available.")

        missing = set(self.features) - set(X.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {list(missing)}")

        # 学習時と同じ順序・構成で特徴量を抽出
        X_processed = X[self.features]

        # 分類タスクの場合、確率を返す（predict_proba）
        if self.task in [MLPTask.BINARY, MLPTask.MULTICLASS]:
            # scikit-learnのpredict_probaは、BINARYの場合も2列（0と1の確率）を返します
            return self.model.predict_proba(X_processed)

        # 回帰タスクの場合は、直接予測値を返す
        elif self.task == MLPTask.REGRESSION:
            return self.model.predict(X_processed)
        else:
             # このコードパスに到達することはないはずですが、念のため
            raise ValueError(f"Unsupported task during prediction: {self.task}")


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

        # 予測値を取得
        # 分類の場合は確率、回帰の場合は予測値本体
        y_pred = self.model.predict(self.X_test)

        metrics = {} # 評価結果を格納する辞書

        # ===== BINARY（2値分類） =====
        if self.task == MLPTask.BINARY:
            # 確率を取得（predict_probaはクラスごとに確率を返す）
            # binaryでは通常、正例（クラス1）の確率を使用
            proba = self.model.predict_proba(self.X_test)[:, 1]

            metrics = {
                MLPMetric.ACCURACY.value: accuracy_score(self.y_test, y_pred),
                MLPMetric.PRECISION.value: precision_score(self.y_test, y_pred),
                MLPMetric.RECALL.value: recall_score(self.y_test, y_pred),
                MLPMetric.F1.value: f1_score(self.y_test, y_pred),
                MLPMetric.AUC.value: roc_auc_score(self.y_test, proba),
                MLPMetric.LOGLOSS.value: log_loss(self.y_test, proba)
            }

        # ===== MULTICLASS（多クラス分類） =====
        elif self.task == MLPTask.MULTICLASS:
            # 多クラス分類の確率は各クラスの確率を含む配列
            proba = self.model.predict_proba(self.X_test)

            metrics = {
                MLPMetric.MULTI_LOGLOSS.value: log_loss(self.y_test, proba) # 多クラス分類では通常loglossを使用
            }
            # 多クラス分類のAUCは計算方法がいくつかあるが、ここではovrを使用
            # クラス数が2以下の場合はエラーになるためtry-exceptで処理
            try:
                # y_testをone-hotエンコードしてroc_auc_scoreに渡す必要がある場合がある
                # sklearnのroc_auc_scoreは、y_trueがラベル、y_scoreがprobaでmulti_class='ovr' or 'ovo'をサポート
                metrics[MLPMetric.AUC.value] = roc_auc_score(self.y_test, proba, multi_class='ovr')
            except ValueError as e:
                # 例: クラスが1種類しかない場合などに発生
                print(f"Warning: Could not compute AUC for multiclass task. {e}")
                metrics[MLPMetric.AUC.value] = np.nan # 計算できない場合はNaNとする

        # ===== REGRESSION（回帰） =====
        elif self.task == MLPTask.REGRESSION:
            mse = mean_squared_error(self.y_test, y_pred)
            metrics = {
                MLPMetric.MSE.value: mse,
                MLPMetric.RMSE.value: np.sqrt(mse), # MSEからRMSEを計算
                MLPMetric.MAE.value: mean_absolute_error(self.y_test, y_pred),
                MLPMetric.R2.value: r2_score(self.y_test, y_pred)
            }
        else:
            # サポートされていないタスク
             raise ValueError(f"Unsupported task during evaluation: {self.task}")


        # 結果をDataFrameに整形して返す
        return pd.DataFrame.from_dict(metrics, orient="index", columns=["Score"])


    # scikit-learnのMLPには標準的な feature_importances_ 属性がないため、このメソッドは省略します。
    # 必要に応じて、Permutation Importanceなどの手法を別途実装することは可能です。
    # def feature_importances(self, top_k: Optional[int] = None) -> pd.DataFrame:
    #     pass # 実装しない、またはエラーを返す


    def cross_validate(
        self,
        cv: Optional[int] = 5
    ) -> pd.DataFrame:
        """
        クロスバリデーションを実行してモデルの性能を評価します。

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
        ValueError
            サポートされていないタスク種別が設定されている場合。
        """
        # データが設定されているかチェック（load_data()の呼び出しが必要）
        if self.df is None or self.target is None or self.features is None:
            raise ValueError("Data is not prepared. Call load_data() first.")

        # 説明変数（X）と目的変数（y）を取得
        X = self.df[self.features]
        y = self.df[self.target]

        # 新たにモデルをインスタンス化（毎回CVごとに同一条件で評価するため）
        # ここで get_model() を再度呼び出すことで、各フォールドで独立したモデルが使われます。
        model = self.get_model()

        scoring_dict: dict[str, str] = {}
        # タスクごとに適切な評価指標（scoring）を指定
        # scikit-learnのscoring文字列を使用
        if self.task == MLPTask.BINARY:
            scoring_dict = {
                MLPMetric.ACCURACY.value: 'accuracy',
                MLPMetric.AUC.value: 'roc_auc',
                MLPMetric.LOGLOSS.value: 'neg_log_loss' # cross_val_scoreでは損失は負の値で返る
            }
        elif self.task == MLPTask.MULTICLASS:
            scoring_dict = {
                 # 多クラスAUCは 'roc_auc_ovr' または 'roc_auc_ovo' を使用
                MLPMetric.AUC.value: 'roc_auc_ovr',
                MLPMetric.MULTI_LOGLOSS.value: 'neg_log_loss'
            }
        elif self.task == MLPTask.REGRESSION:
            scoring_dict = {
                MLPMetric.MSE.value: 'neg_mean_squared_error',
                MLPMetric.MAE.value: 'neg_mean_absolute_error',
                MLPMetric.R2.value: 'r2'
            }
        else:
             raise ValueError(f"Unsupported task for cross-validation: {self.task}")

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

                # MSEからRMSEを手動で計算して追加（回帰タスクのみ）
                if self.task == MLPTask.REGRESSION and metric_name == MLPMetric.MSE.value:
                    results[MLPMetric.RMSE.value] = np.sqrt(scores)

            except Exception as e:
                # クロスバリデーション実行中にエラーが発生した場合
                print(f"Error during cross-validation for scoring '{scoring_method}': {str(e)}")
                # エラーが発生したメトリックの結果はNaNなどとするか、処理を停止するか選択
                # ここではエラーメッセージを出力し、そのまま続行（他のメトリックは計算を試みる）
                # より堅牢にするなら、エラーをraiseするか、このメトリックをスキップする処理を入れる
                results[metric_name] = np.full(cv, np.nan) # エラー時はNaNの配列で埋める

        # 結果をDataFrame形式で返す（各列が評価指標、各行がCVの分割結果）
        return pd.DataFrame(results)
