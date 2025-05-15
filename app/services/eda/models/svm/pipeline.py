import pandas as pd
import numpy as np
from typing import Optional, Union, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, mean_squared_error,
    mean_absolute_error, r2_score
)
from sklearn.svm import SVC, SVR
from .enum import SVMTask, SVMMetric
from .parameter import get_svm_params


class SVMPipeline:
    """
    サポートベクターマシン (SVM - SVC/SVR) モデルを使用したパイプライン。
    データの読み込み、学習用/テスト用分割、モデル学習、予測、評価、係数/サポートベクター、クロスバリデーションをサポート。
    """
    def __init__(self, task: Optional[SVMTask] = SVMTask.CLASSIFICATION):
        """
        SVMPipelineを初期化します。

        Parameters
        ----------
        task : Optional[SVMTask], default=SVMTask.CLASSIFICATION
            パイプラインのタスク種別 (CLASSIFICATION, REGRESSION)。
        """
        if task not in [SVMTask.CLASSIFICATION, SVMTask.REGRESSION]:
             raise ValueError("SVM only supports CLASSIFICATION or REGRESSION task.")

        self.task: SVMTask = task
        # random_stateはモデルパラメータだが、パイプラインでも保持
        self._random_state: int = 42 # デフォルト値
        self.params: dict = self.get_params()
        # モデルはタスクとパラメータに基づいて get_model() で初期化
        self.model: Union[SVC, SVR] = self.get_model()

        # データおよび分割されたデータを保持するための属性
        self.features: Optional[list[str]] = None  # 入力特徴量の名前のリスト
        self.target: Optional[str] = None  # 目的変数のカラム名
        self.df: Optional[pd.DataFrame] = None  # ロードされた元のDataFrame
        self.X_train: Optional[pd.DataFrame] = None  # 特徴量の学習データ
        self.X_test: Optional[pd.DataFrame] = None  # 特徴量のテストデータ
        self.y_train: Optional[pd.Series] = None  # 目的変数の学習データ
        self.y_test: Optional[pd.Series] = None  # 目的変数のテストデータ


    def load_task(self, task: SVMTask) -> None:
        """
        パイプラインのタスク種別を変更し、パラメータとモデルを更新します。

        Parameters
        ----------
        task : SVMTask
            新しいタスク種別 (CLASSIFICATION, REGRESSION)。

        Raises
        ------
        ValueError
            サポートされていないタスクが指定された場合。
        """
        if task not in [SVMTask.CLASSIFICATION, SVMTask.REGRESSION]:
             raise ValueError("SVM only supports CLASSIFICATION or REGRESSION task.")

        self.task = task
        self.params = self.get_params() # 新しいタスクのデフォルトパラメータを取得
        # random_state は get_params の結果に含まれているはずなので、ここでは特別扱いは不要
        self.model = self.get_model()   # 新しいモデルを初期化


    def get_params(self) -> dict:
        """
        現在のタスク種別に基づいてデフォルトパラメータを取得します。

        Returns
        -------
        dict
            モデルのパラメータ辞書。
        """
        # パイプラインの random_state を get_svm_params に渡して、モデルパラメータに含める
        params = get_svm_params(self.task)
        params['random_state'] = self._random_state # パイプラインで保持しているrandom_stateを使用
        return params


    def get_model(self) -> Union[SVC, SVR]:
        """
        現在のタスク種別とパラメータに基づいてSVMモデル (SVC/SVR) を初期化します。

        Returns
        -------
        Union[SVC, SVR]
            初期化されたSVMモデルインスタンス。

        Raises
        ------
        ValueError
            サポートされていないタスクが設定されている場合。
        """
        # get_params() を通じて取得したパラメータを使用
        params = self.get_params() # get_params() は _random_state を含んだdictを返す
        if self.task == SVMTask.CLASSIFICATION:
            return SVC(**params)
        elif self.task == SVMTask.REGRESSION:
            return SVR(**params)
        else:
            raise ValueError(f"Unsupported task for SVM model: {self.task}")


    def load_data(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str,
        test_size: Optional[float] = 0.2,
        random_state: Optional[int] = 42 # load_dataに渡されたrandom_stateをパイプラインで保持
    ) -> None:
        """
        指定された特徴量と目的変数をもとに、DataFrameを学習用とテスト用データに分割します。
        SVMは特徴量のスケールに非常に敏感です。

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
            分類タスクなのにターゲットカラムが数値型でない場合、または回帰タスクなのにターゲットカラムが数値型でない場合。
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

        # タスクに応じてターゲットの数値型チェック
        if self.task == SVMTask.CLASSIFICATION:
            # SVCは数値ラベル（int/float）を期待します
            if not pd.api.types.is_numeric_dtype(y):
                 raise TypeError(
                     f"Target column '{target}' is of type {y.dtype} (not numeric). "
                     f"SVC requires numerical labels. Please encode the labels as integers (e.g., using LabelEncoder) before calling load_data()."
                 )
            # 分類タスクでは層化抽出を行う
            stratify = y
        elif self.task == SVMTask.REGRESSION:
            # SVRは数値ターゲット（int/float）を期待します
            if not pd.api.types.is_numeric_dtype(y):
                 raise TypeError(
                     f"Target column '{target}' is of type {y.dtype} (not numeric). "
                     f"SVR requires numerical targets. Please ensure the target column is numeric before calling load_data()."
                 )
            # 回帰タスクでは層化抽出は通常行いません
            stratify = None
        else:
             # このパイプラインは分類/回帰専用なので、ここに到達することはないはず
             raise ValueError(f"Unsupported task during load_data: {self.task}")


        # 学習データとテストデータに分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self._random_state, stratify=stratify
        )


    def fit(self) -> None:
        """
        学習データ (self.X_train, self.y_train) を用いてSVMモデルを訓練します。
        SVMは大規模データで訓練に時間がかかる場合があります。
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call load_data() first.")

        # SVMモデルの fit メソッドを呼び出す
        self.model.fit(self.X_train, self.y_train)


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        学習済みのSVMモデルを用いて、新しいデータに対する予測を行います。

        Parameters
        ----------
        X : pd.DataFrame
            予測に使用する特徴量データ。
            モデルを学習した際に指定した self.features と同じカラム構成である必要があります。

        Returns
        -------
        np.ndarray
            予測結果。
            - タスクが CLASSIFICATION かつモデルが確率推定可能な場合は、各クラスに対する確率を含む 2次元配列（predict_proba）。
            - それ以外の場合（CLASSIFICATIONで確率推定不可、REGRESSION）は、予測値の 1次元配列（predict）。

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

        if self.task == SVMTask.CLASSIFICATION:
            # SVCで確率推定が有効な場合のみ predict_proba を使用
            if hasattr(self.model, 'predict_proba'):
                 return self.model.predict_proba(X_processed)
            else:
                 # 確率推定が有効でない場合 (probability=False)、predict()の結果を返す
                 # evaluate メソッドなどで確率が必要な場合は、probability=True で訓練する必要があります。
                 # ここでは predict() の結果をそのまま返しますが、predict() はクラスラベルを返す点に注意。
                 # 評価指標計算では predict_proba が必要になるため、evaluate側で別途 predict_proba を試行します。
                 return self.model.predict(X_processed)

        elif self.task == SVMTask.REGRESSION:
            # SVRは predict_proba は持っていません
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
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not available. Call load_data() with test_size > 0.")

        # 予測ラベルまたは回帰値を取得
        # Classificationの場合、predict() はクラスラベルを返します
        y_pred = self.model.predict(self.X_test)

        metrics = {} # 評価結果を格納する辞書

        # --- CLASSIFICATION（分類） ---
        if self.task == SVMTask.CLASSIFICATION:
            # 二値分類か多クラス分類かを判定
            is_binary = self.y_test.nunique() <= 2

            # 共通評価指標
            metrics[SVMMetric.ACCURACY.value] = accuracy_score(self.y_test, y_pred)

            # Precision, Recall, F1 は二値か多クラスかで average パラメータが必要
            average_method = 'binary' if is_binary else 'weighted'
            try:
                metrics[SVMMetric.PRECISION.value] = precision_score(self.y_test, y_pred, average=average_method)
                metrics[SVMMetric.RECALL.value] = recall_score(self.y_test, y_pred, average=average_method)
                metrics[SVMMetric.F1.value] = f1_score(self.y_test, y_pred, average=average_method)
            except ValueError:
                 # エラーが発生した場合はNaNとする
                 metrics[SVMMetric.PRECISION.value] = np.nan
                 metrics[SVMMetric.RECALL.value] = np.nan
                 metrics[SVMMetric.F1.value] = np.nan

            # AUC と LogLoss は predict_proba が必要です (SVC初期化時に probability=True が必要)
            if hasattr(self.model, 'predict_proba'):
                 y_pred_proba = self.model.predict_proba(self.X_test)
                 if is_binary:
                     proba_for_binary_metrics = y_pred_proba[:, 1]
                     metrics[SVMMetric.AUC.value] = roc_auc_score(self.y_test, proba_for_binary_metrics)
                     metrics[SVMMetric.LOGLOSS.value] = log_loss(self.y_test, proba_for_binary_metrics)
                 else:
                     metrics[SVMMetric.LOGLOSS.value] = log_loss(self.y_test, y_pred_proba)
                     # 多クラスAUC (ovr)
                     try:
                          metrics[SVMMetric.AUC.value] = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr')
                     except ValueError:
                          metrics[SVMMetric.AUC.value] = np.nan # 計算できない場合はNaN
            else:
                 # probability=False の場合、AUCとLogLossは計算できません
                 metrics[SVMMetric.AUC.value] = np.nan
                 metrics[SVMMetric.LOGLOSS.value] = np.nan


        # --- REGRESSION（回帰） ---
        elif self.task == SVMTask.REGRESSION:
            mse = mean_squared_error(self.y_test, y_pred)
            metrics = {
                SVMMetric.MSE.value: mse,
                SVMMetric.RMSE.value: np.sqrt(mse), # MSEからRMSEを計算
                SVMMetric.MAE.value: mean_absolute_error(self.y_test, y_pred),
                SVMMetric.R2.value: r2_score(self.y_test, y_pred)
            }

        else:
             raise ValueError(f"Unsupported task during evaluation: {self.task}")


        return pd.DataFrame.from_dict(metrics, orient="index", columns=["Score"])


    def feature_coefficients(self) -> Optional[pd.DataFrame]:
        """
        学習済みモデルから特徴量の係数を取得します。
        リニアカーネル (kernel='linear') の場合にのみ係数が利用可能です。
        非リニアカーネルの場合はNoneを返します。

        Returns
        -------
        pd.DataFrame or None
            リニアカーネルの場合は特徴量名と係数を含むDataFrame。
            非リニアカーネルの場合はNone。

        Raises
        ------
        ValueError
            モデルが訓練されていない、または必要な属性がない場合。
        """
        if self.model is None or not hasattr(self.model, "coef_"):
            # coef_ は fit が成功すれば存在するはずですが、カーネルによっては利用できません。
            # リニアカーネル以外の場合は AttributeError が発生します。
            # ここでは、modelが訓練されていない場合も捕捉します。
             raise ValueError("モデルが訓練されていないか、係数情報が利用できません。")

        # coef_ 属性は kernel='linear' の場合にのみ存在します
        if not hasattr(self.model, "coef_"):
            # 非リニアカーネルの場合
            # print("Feature coefficients are only available for kernel='linear'.")
            return None # 非リニアカーネルの場合はNoneを返す

        if self.features is None:
             raise ValueError("特徴量リストが利用できません。load_data()が実行されていませんか？")

        # coef_ の形状は SVC/SVR の multi_class や multi-target によって異なります。
        # ここでは単一ターゲットの分類/回帰を想定し、coef_ が (1, n_features) または (n_features,) の形状であることを期待します。
        # SVCの多クラス分類で multi_class='ovr' の場合、(n_classes, n_features) になります。
        # 簡単のため、coef_ の最初の次元が1であるか、または1次元配列であることを想定します。
        # それ以外の場合は NotImplementedError とします。

        coefficients = self.model.coef_

        if coefficients.ndim > 1 and coefficients.shape[0] > 1:
             # 多クラス分類 ('ovr') などでクラスごとの係数になっている場合
             # この表示形式（特徴量ごとの単一の棒グラフ）には向かないため、一旦エラーとします。
             raise NotImplementedError(f"Multi-class or multi-output coefficients with shape {coefficients.shape} display is not yet supported in this format.")

        # 二値分類、または multi_class='ovr' で単一クラスの係数、または回帰の場合
        # coef_ は (1, n_features) または (n_features,) の形状になるはず
        if coefficients.ndim > 1:
            coefficients = coefficients[0] # (1, n_features) の場合は最初の行を取得

        # 特徴量名と係数をDataFrameにまとめる
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
        if self.df is None or self.target is None or self.features is None:
            raise ValueError("Data is not prepared. Call load_data() first.")

        X = self.df[self.features]
        y = self.df[self.target]

        # 新たにモデルをインスタンス化（毎回CVごとに同一条件で評価するため）
        # self.get_model() を呼び出すことで、self.params が反映されたモデルが作成されます。
        model = self.get_model()

        # scikit-learnのscoring文字列を定義
        scoring_dict: dict[str, str] = {}

        if self.task == SVMTask.CLASSIFICATION:
            # 分類タスクの評価指標
            scoring_dict = {
                SVMMetric.ACCURACY.value: 'accuracy',
                SVMMetric.LOGLOSS.value: 'neg_log_loss' # 損失は負の値で返る
                # AUCは probability=True が必要
            }
            # probability=True かつ二値分類なら roc_auc, 多クラスなら roc_auc_ovr
            # CVの内部では毎回新しいモデルが作成されるため、probabilityの設定はmodelに依存します。
            # model.get_params() で probability を確認できます。
            # scikit-learnのcross_val_scoreは 'roc_auc' や 'neg_log_loss' を使う場合、
            # 推定器が predict_proba を持つことを期待します。
            # モデルのパラメータで probability=True に設定されている必要があります。
            model_params = model.get_params()
            if model_params.get('probability', False): # probability が True に設定されているか確認
                 # y.nunique() は CV の分割ごとに変わる可能性があるが、全体のyで判定
                 if y.nunique() <= 2:
                      scoring_dict[SVMMetric.AUC.value] = 'roc_auc' # 二値分類用
                 else:
                      scoring_dict[SVMMetric.AUC.value] = 'roc_auc_ovr' # 多クラス分類用 ovr

            else:
                 # probability=False の場合、AUCとLogLossはCVで計算できません
                 # 警告は evaluate メソッド側で表示します
                 pass # スコアリング辞書に追加しない


        elif self.task == SVMTask.REGRESSION:
            # 回帰タスクの評価指標
            scoring_dict = {
                SVMMetric.MSE.value: 'neg_mean_squared_error', # MSEは負の値で返る
                SVMMetric.MAE.value: 'neg_mean_absolute_error', # MAEは負の値で返る
                SVMMetric.R2.value: 'r2'
            }
        else:
             raise ValueError(f"Unsupported task for cross-validation: {self.task}")

        results: dict[str, Any] = {} # 各評価指標のCVスコアを格納する辞書

        # 指定された全ての評価指標でcross_val_scoreを実行
        for metric_name, scoring_method in scoring_dict.items():
            try:
                # cross_val_score を実行
                # Classificationタスクで probability=False の場合に
                # 'roc_auc' や 'neg_log_loss' を指定するとエラーになります。
                # scoring_dictに追加する前にprobabilityチェックを入れるか、
                # エラーを捕捉してNaNで埋める必要があります。
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

                # 回帰タスクでMSEの場合、RMSEを手動で計算して追加
                if self.task == SVMTask.REGRESSION and metric_name == SVMMetric.MSE.value:
                    results[SVMMetric.RMSE.value] = np.sqrt(scores)


            except Exception as e:
                print(f"Warning: Could not compute cross-validation score for metric '{metric_name}' using scoring '{scoring_method}'. Error: {str(e)}")
                # エラーが発生したメトリックの結果はNaNなどで埋める
                results[metric_name] = np.full(cv, np.nan) # エラー時はNaNの配列で埋める


        # 結果をDataFrame形式で返す（各列が評価指標、各行がCVの分割結果）
        return pd.DataFrame(results)