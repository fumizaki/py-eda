import pandas as pd
import numpy as np
from typing import Optional, Union, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, mean_squared_error,
    mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .enum import RFTask, RFMetric
from .parameter import get_rf_params


class RandomForestPipeline:
    """
    Random Forestモデルを使用した機械学習パイプライン。
    データの読み込み、学習用/テスト用分割、モデル学習、予測、評価、特徴量重要度、クロスバリデーションをサポート。
    """
    def __init__(self, task: Optional[RFTask] = RFTask.CLASSIFICATION):
        """
        RandomForestPipelineを初期化します。

        Parameters
        ----------
        task : Optional[RFTask], default=RFTask.CLASSIFICATION
            パイプラインのタスク種別 (分類、回帰)。
        """
        self.task: RFTask = task
        self.params: dict = self.get_params()
        # モデルはタスクとパラメータに基づいて get_model() で初期化
        self.model: Union[RandomForestClassifier, RandomForestRegressor] = self.get_model()

        # データおよび分割されたデータを保持するための属性
        self.features: Optional[list[str]] = None  # 入力特徴量の名前のリスト
        self.target: Optional[str] = None  # 目的変数のカラム名
        self.df: Optional[pd.DataFrame] = None  # ロードされた元のDataFrame
        self.X_train: Optional[pd.DataFrame] = None  # 特徴量の学習データ
        self.X_test: Optional[pd.DataFrame] = None  # 特徴量のテストデータ
        self.y_train: Optional[pd.Series] = None  # 目的変数の学習データ
        self.y_test: Optional[pd.Series] = None  # 目的変数のテストデータ


    def load_task(self, task: RFTask) -> None:
        """
        パイプラインのタスク種別を変更し、パラメータとモデルを更新します。

        Parameters
        ----------
        task : RFTask
            新しいタスク種別。
        """
        self.task = task
        self.params = self.get_params() # 新しいタスクのデフォルトパラメータを取得
        self.model = self.get_model()   # 新しいモデルを初期化


    def get_params(self) -> dict:
        """
        現在のタスク種別に基づいてデフォルトパラメータを取得します。

        Returns
        -------
        dict
            モデルのパラメータ辞書。
        """
        return get_rf_params(self.task)


    def get_model(self) -> Union[RandomForestClassifier, RandomForestRegressor]:
        """
        現在のタスク種別とパラメータに基づいてRandom Forestモデルを初期化します。

        Returns
        -------
        Union[RandomForestClassifier, RandomForestRegressor]
            初期化されたRandom Forestモデルインスタンス。

        Raises
        ------
        ValueError
            サポートされていないタスク種別が設定されている場合。
        """
        if self.task == RFTask.CLASSIFICATION:
            # 分類モデルを初期化
            return RandomForestClassifier(**self.params)
        elif self.task == RFTask.REGRESSION:
            # 回帰モデルを初期化
            return RandomForestRegressor(**self.params)
        else:
            raise ValueError(f"Unsupported task for Random Forest model: {self.task}")


    def load_data(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str,
        test_size: Optional[float] = 0.2,
        random_state: Optional[int] = 42
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
             raise ValueError(f"Input DataFrame is missing required features: {list(missing_features)}")

        # 属性として保持
        self.df = df
        self.features = features
        self.target = target

        # 特徴量と目的変数を抽出
        X = df[features]
        y = df[target]

        # 分類タスクかつ目的変数が数値型でない場合のエラーチェック
        # RandomForestClassifierは数値ラベルまたは文字列ラベルを直接扱えますが、
        # 評価指標の計算（特にroc_auc_scoreなど）やクロスバリデーションでは数値ラベルが前提となるため、
        # 事前に数値エンコーディングされていることを推奨・要求します。
        if self.task == RFTask.CLASSIFICATION and not pd.api.types.is_numeric_dtype(y):
             # ただし、数値エンコーディングされていない文字列ラベルでもfitは可能だが、evaluateで問題が起こる可能性がある
             # 強制的に数値エンコーディングを要求する
             raise TypeError(
                 f"Target column '{target}' is of type {y.dtype} (not numeric). "
                 f"For classification tasks with RandomForest, please encode the labels as integers (e.g., using LabelEncoder) before calling load_data()."
             )


        # 分類タスクでは層化抽出を行う（データ分割の際にクラス比率を維持）
        stratify = y if self.task == RFTask.CLASSIFICATION else None

        # 学習データとテストデータに分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )


    def fit(self) -> None:
        """
        学習データ (self.X_train, self.y_train) を用いてRandom Forestモデルを訓練します。
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call load_data() first.")

        # Random Forestモデルの fit メソッドを呼び出す
        self.model.fit(self.X_train, self.y_train)


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        学習済みのRandom Forestモデルを用いて、新しいデータに対する予測を行います。

        Parameters
        ----------
        X : pd.DataFrame
            予測に使用する特徴量データ。
            モデルを学習した際に指定した self.features と同じカラム構成である必要があります。

        Returns
        -------
        np.ndarray
            予測結果。
            - タスクが CLASSIFICATION の場合は、各クラスに対する確率を含む 2次元配列（predict_proba）。
            - タスクが REGRESSION の場合は、予測値の 1次元配列（predict）。

        Raises
        ------
        ValueError
            入力データに、モデルが必要とする特徴量（self.features）の一部が欠けている場合。
        ValueError
            分類タスクでpredict_probaが利用できない場合（発生しないはず）。
        """
        if self.features is None:
             raise ValueError("Model has not been trained yet. Feature list is not available.")

        missing = set(self.features) - set(X.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {list(missing)}")

        X_processed = X[self.features]

        if self.task == RFTask.CLASSIFICATION:
            # 分類タスクの場合、確率を返す
            # RandomForestClassifier は predict_proba を持っています
            if hasattr(self.model, 'predict_proba'):
                 return self.model.predict_proba(X_processed)
            else:
                 # 理論上RandomForestClassifierでは起こりえないが、念のため
                 raise ValueError("Classifier model does not support predict_proba.")

        elif self.task == RFTask.REGRESSION:
            # 回帰タスクの場合、直接予測値を返す
            return self.model.predict(X_processed)
        else:
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
        ValueError
            サポートされていないタスク種別が設定されている場合。
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not available. Call load_data() with test_size > 0.")

        # 予測値を取得 (クラスラベル または 回帰値)
        y_pred = self.model.predict(self.X_test)

        metrics = {} # 評価結果を格納する辞書

        # --- CLASSIFICATION（分類） ---
        if self.task == RFTask.CLASSIFICATION:
            # 確率を取得（評価指標によっては必要）
            proba = self.predict(self.X_test) # predict()を再利用して確率を取得

            # 二値分類の場合は正例（クラス1）の確率、多クラス分類の場合は全クラスの確率
            if proba.shape[1] == 2: # 二値分類と判定
                 proba_for_binary_metrics = proba[:, 1]
                 metrics[RFMetric.AUC.value] = roc_auc_score(self.y_test, proba_for_binary_metrics)
                 metrics[RFMetric.LOGLOSS.value] = log_loss(self.y_test, proba_for_binary_metrics)
            else: # 多クラス分類
                 metrics[RFMetric.MULTI_LOGLOSS.value] = log_loss(self.y_test, proba)
                 # 多クラスAUCは計算方法がいくつかあるため、ここではovrを使用（クラス数が2以下の場合はエラー）
                 try:
                     metrics[RFMetric.AUC.value] = roc_auc_score(self.y_test, proba, multi_class='ovr')
                 except ValueError:
                     metrics[RFMetric.AUC.value] = np.nan # 計算できない場合はNaN

            # クラスラベルを使った共通指標
            metrics[RFMetric.ACCURACY.value] = accuracy_score(self.y_test, y_pred)
            # precision, recall, f1 は二値分類または多クラス分類で average パラメータが必要
            # ここでは二値分類を想定したweighted averageか、タスクに合わせてaverageを変更する必要がありますが、
            # 簡単のため二値分類のデフォルト (binary) で計算できる場合とします。
            # 多クラス分類の場合、average='weighted'などを指定することが多いです。
            # 例：precision_score(self.y_test, y_pred, average='weighted')
            # ここでは最も単純な二値分類のデフォルト（pos_label=1）または多クラスのエラー回避のため try-except を使用
            try:
                metrics[RFMetric.PRECISION.value] = precision_score(self.y_test, y_pred)
                metrics[RFMetric.RECALL.value] = recall_score(self.y_test, y_pred)
                metrics[RFMetric.F1.value] = f1_score(self.y_test, y_pred)
            except ValueError:
                # 多クラス分類などでエラーになる場合があるためNaNとする
                metrics[RFMetric.PRECISION.value] = np.nan
                metrics[RFMetric.RECALL.value] = np.nan
                metrics[RFMetric.F1.value] = np.nan


        # --- REGRESSION（回帰） ---
        elif self.task == RFTask.REGRESSION:
            mse = mean_squared_error(self.y_test, y_pred)
            metrics = {
                RFMetric.MSE.value: mse,
                RFMetric.RMSE.value: np.sqrt(mse), # MSEからRMSEを計算
                RFMetric.MAE.value: mean_absolute_error(self.y_test, y_pred),
                RFMetric.R2.value: r2_score(self.y_test, y_pred)
            }

        else:
             raise ValueError(f"Unsupported task during evaluation: {self.task}")


        return pd.DataFrame.from_dict(metrics, orient="index", columns=["Score"])


    def feature_importances(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        モデルから特徴量の重要度を取得します。

        Parameters
        ----------
        top_k : int, optional
            上位k件の特徴量を取得します。Noneの場合は全て取得します。

        Returns
        -------
        pd.DataFrame
            特徴量名と重要度を含むDataFrame。

        Raises
        ------
        ValueError
            モデルが訓練されていない、または特徴量重要度情報が取得できない場合。
        """
        # RandomForestモデルは feature_importances_ 属性を持っています
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            raise ValueError("モデルが訓練されていないか、特徴量重要度情報が利用できません。")
        if self.features is None:
             raise ValueError("特徴量リストが利用できません。load_data()が実行されていませんか？")


        importances = self.model.feature_importances_  # Random Forestによる特徴量の重要度

        # 特徴量名と重要度をDataFrameにまとめる
        importance_df = pd.DataFrame({
            "feature": self.features,
            "importance": importances
        }).sort_values(by="importance", ascending=False) # 重要度で降順にソート

        # top_kが指定されていれば上位を抽出
        if top_k is not None:
            importance_df = importance_df.head(top_k)

        return importance_df.reset_index(drop=True)


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
        # self.get_model() を呼び出すことで、self.params が反映されたモデルが作成されます。
        model = self.get_model()

        # scikit-learnのscoring文字列を定義
        scoring_dict: dict[str, str] = {}
        if self.task == RFTask.CLASSIFICATION:
            # 分類タスクの評価指標
            scoring_dict = {
                RFMetric.ACCURACY.value: 'accuracy',
                RFMetric.AUC.value: 'roc_auc' if y.nunique() > 2 else 'roc_auc', # 二値/多クラスでroc_aucの計算が異なる場合があるが、sklearnは自動判別
                RFMetric.LOGLOSS.value: 'neg_log_loss' # 損失は負の値で返る
                # 多クラス分類の precision, recall, f1 には 'average' パラメータが必要なため、
                # cross_val_scoreでは単純な文字列指定が難しい場合があります。
                # 必要なら make_scorer を使うか、evaluate メソッドで詳細を見るように誘導します。
                # ここではシンプルに一般的な指標のみ含めます。
            }
            # 二値分類のみの場合、average='binary' (デフォルト) で計算可能
            if y.nunique() <= 2:
                 scoring_dict[RFMetric.PRECISION.value] = 'precision'
                 scoring_dict[RFMetric.RECALL.value] = 'recall'
                 scoring_dict[RFMetric.F1.value] = 'f1'


        elif self.task == RFTask.REGRESSION:
            # 回帰タスクの評価指標
            scoring_dict = {
                RFMetric.MSE.value: 'neg_mean_squared_error', # MSEは負の値で返る
                RFMetric.MAE.value: 'neg_mean_absolute_error', # MAEは負の値で返る
                RFMetric.R2.value: 'r2'
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

                # MSE から RMSE を手動で計算して追加（回帰タスクのみ）
                if self.task == RFTask.REGRESSION and metric_name == RFMetric.MSE.value:
                    results[RFMetric.RMSE.value] = np.sqrt(scores)

            except Exception as e:
                print(f"Warning: Could not compute cross-validation score for metric '{metric_name}' using scoring '{scoring_method}'. Error: {str(e)}")
                # エラーが発生したメトリックの結果はNaNなどで埋める
                results[metric_name] = np.full(cv, np.nan) # エラー時はNaNの配列で埋める


        # 結果をDataFrame形式で返す（各列が評価指標、各行がCVの分割結果）
        return pd.DataFrame(results)