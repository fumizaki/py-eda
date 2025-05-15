import pandas as pd
import numpy as np
from typing import Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)
from sklearn.linear_model import LogisticRegression
from .enum import LogiRegTask, LogiRegMetric
from .parameter import get_logireg_params


class LogisticRegressionPipeline:
    """
    ロジスティック回帰 (Logistic Regression) モデルを使用した分類パイプライン。
    データの読み込み、学習用/テスト用分割、モデル学習、予測、評価、係数、クロスバリデーションをサポート。
    """
    def __init__(self, task: Optional[LogiRegTask] = LogiRegTask.CLASSIFICATION):
        """
        LogisticRegressionPipelineを初期化します。

        Parameters
        ----------
        task : Optional[LogiRegTask], default=LogiRegTask.CLASSIFICATION
            パイプラインのタスク種別 (CLASSIFICATION)。
        """
        if task != LogiRegTask.CLASSIFICATION:
             raise ValueError("Logistic Regression only supports CLASSIFICATION task.")

        self.task: LogiRegTask = task
        self.params: dict = self.get_params()
        # モデルはタスクとパラメータに基づいて get_model() で初期化
        self.model: LogisticRegression = self.get_model() # ロジスティック回帰はClassifierのみ

        # データおよび分割されたデータを保持するための属性
        self.features: Optional[list[str]] = None  # 入力特徴量の名前のリスト
        self.target: Optional[str] = None  # 目的変数のカラム名
        self.df: Optional[pd.DataFrame] = None  # ロードされた元のDataFrame
        self.X_train: Optional[pd.DataFrame] = None  # 特徴量の学習データ
        self.X_test: Optional[pd.DataFrame] = None  # 特徴量のテストデータ
        self.y_train: Optional[pd.Series] = None  # 目的変数の学習データ
        self.y_test: Optional[pd.Series] = None  # 目的変数のテストデータ


    def load_task(self, task: LogiRegTask) -> None:
        """
        パイプラインのタスク種別を変更し、パラメータとモデルを更新します。

        Parameters
        ----------
        task : LogiRegTask
            新しいタスク種別。Logistic RegressionではCLASSIFICATIONのみ有効。

        Raises
        ------
        ValueError
            CLASSIFICATION以外のタスクが指定された場合。
        """
        if task != LogiRegTask.CLASSIFICATION:
             raise ValueError("Logistic Regression only supports CLASSIFICATION task.")

        self.task = task
        self.params = self.get_params() # 新しいタスクのデフォルトパラメータを取得 (CLASSIFICATION用)
        self.model = self.get_model()   # 新しいモデルを初期化


    def get_params(self) -> dict:
        """
        現在のタスク種別に基づいてデフォルトパラメータを取得します。

        Returns
        -------
        dict
            モデルのパラメータ辞書。
        """
        # Logistic RegressionはCLASSIFICATIONタスクのみを想定してパラメータを取得
        return get_logireg_params(LogiRegTask.CLASSIFICATION)


    def get_model(self) -> LogisticRegression:
        """
        現在のタスク種別とパラメータに基づいてLogistic Regressionモデルを初期化します。

        Returns
        -------
        LogisticRegression
            初期化されたLogistic Regressionモデルインスタンス。
        """
        # Logistic Regressionは常にClassifier
        return LogisticRegression(**self.params)


    def load_data(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str,
        test_size: Optional[float] = 0.2,
        random_state: Optional[int] = 42
    ) -> None:
        """
        分類用のデータを学習用とテスト用データに分割します。

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
            再現性を確保するための乱数シード。

        Raises
        ------
        ValueError
            指定されたターゲットカラムまたは特徴量カラムの一部がDataFrameに存在しない場合。
        TypeError
            ターゲットカラムが数値型（integerまたはfloat）でない場合（Logistic Regressionは数値ラベルを期待するため）。
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

        # ロジスティック回帰は数値ラベル（int/float）を期待します
        if not pd.api.types.is_numeric_dtype(y):
             raise TypeError(
                 f"Target column '{target}' is of type {y.dtype} (not numeric). "
                 f"Logistic Regression requires numerical labels. Please encode the labels as integers (e.g., using LabelEncoder) before calling load_data()."
             )

        # 分類タスクでは層化抽出を行う（データ分割の際にクラス比率を維持）
        # Logistic Regressionは分類専用なので、常にstratifyを使用
        stratify = y

        # 学習データとテストデータに分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )


    def fit(self) -> None:
        """
        学習データ (self.X_train, self.y_train) を用いてLogistic Regressionモデルを訓練します。
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call load_data() first.")

        # LogisticRegressionモデルの fit メソッドを呼び出す
        self.model.fit(self.X_train, self.y_train)


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        学習済みのLogistic Regressionモデルを用いて、新しいデータに対する予測を行います。

        Parameters
        ----------
        X : pd.DataFrame
            予測に使用する特徴量データ。
            モデルを学習した際に指定した self.features と同じカラム構成である必要があります。

        Returns
        -------
        np.ndarray
            予測結果（確率）。Classificationタスクなので predict_proba() の結果を返します。

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

        # ロジスティック回帰は確率を返す predict_proba を持っています
        return self.model.predict_proba(X_processed)


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

        # 予測ラベルを取得 (確率ではなく最終的なクラス予測)
        y_pred_label = self.model.predict(self.X_test)

        # 予測確率を取得 (AUC, LogLossに必要)
        y_pred_proba = self.predict(self.X_test) # predict() を再利用して確率を取得

        metrics = {} # 評価結果を格納する辞書

        # ロジスティック回帰は分類専用
        if self.task == LogiRegTask.CLASSIFICATION:
            # 二値分類か多クラス分類かを判定
            is_binary = self.y_test.nunique() <= 2

            # 共通評価指標
            metrics[LogiRegMetric.ACCURACY.value] = accuracy_score(self.y_test, y_pred_label)

            # Precision, Recall, F1 は二値か多クラスかで average パラメータが必要
            # 多クラスの場合は average='weighted' を使用するのが一般的
            average_method = 'binary' if is_binary else 'weighted'
            try:
                metrics[LogiRegMetric.PRECISION.value] = precision_score(self.y_test, y_pred_label, average=average_method)
                metrics[LogiRegMetric.RECALL.value] = recall_score(self.y_test, y_pred_label, average=average_method)
                metrics[LogiRegMetric.F1.value] = f1_score(self.y_test, y_pred_label, average=average_method)
            except ValueError:
                 # エラーが発生した場合はNaNとする (例: クラスが1つしかない場合など)
                 metrics[LogiRegMetric.PRECISION.value] = np.nan
                 metrics[LogiRegMetric.RECALL.value] = np.nan
                 metrics[LogiRegMetric.F1.value] = np.nan


            # AUC と LogLoss
            if is_binary:
                # 二値分類の場合は正例（クラス1）の確率を使用
                proba_for_binary_metrics = y_pred_proba[:, 1]
                metrics[LogiRegMetric.AUC.value] = roc_auc_score(self.y_test, proba_for_binary_metrics)
                metrics[LogiRegMetric.LOGLOSS.value] = log_loss(self.y_test, proba_for_binary_metrics)
            else:
                # 多クラス分類の場合
                metrics[LogiRegMetric.LOGLOSS.value] = log_loss(self.y_test, y_pred_proba)
                # 多クラスAUCは計算方法がいくつかあるため、ここではovrを使用
                try:
                     metrics[LogiRegMetric.AUC.value] = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr')
                except ValueError:
                     metrics[LogiRegMetric.AUC.value] = np.nan # 計算できない場合はNaN


        else:
             # このパイプラインは分類専用なので、ここに到達することはないはず
             raise ValueError(f"Unsupported task during evaluation: {self.task}")


        return pd.DataFrame.from_dict(metrics, orient="index", columns=["Score"])


    def feature_coefficients(self) -> pd.DataFrame:
        """
        学習済みモデルから各特徴量の係数を取得します。
        ロジスティック回帰の係数は特徴量の重要度（寄与度）として解釈できます。

        Returns
        -------
        pd.DataFrame
            特徴量名と係数を含むDataFrame。

        Raises
        ------
        ValueError
            モデルが訓練されていない、または係数情報が取得できない場合。
        NotImplementedError
            多クラス分類（'multi_class'='multinomial'）の場合、係数の形式が異なるため未対応。
        """
        if self.model is None or not hasattr(self.model, "coef_") or not hasattr(self.model, "intercept_"):
            raise ValueError("モデルが訓練されていないか、係数情報が利用できません。")
        if self.features is None:
             raise ValueError("特徴量リストが利用できません。load_data()が実行されていませんか？")

        # ロジスティック回帰の係数 (coef_) と切片 (intercept_)
        # coef_ の形状は (n_classes - 1, n_features) または (1, n_features)
        # intercept_ の形状は (n_classes - 1,) または (1,)

        # 多クラス分類で multi_class='multinomial' の場合、coef_ の形状が異なります。
        # Simplification: 現在は multi_class='auto' を使用しており、
        # デフォルトソルバー('lbfgs','newton-cg','sag')と組み合わせると
        # 二値は(1, n_features)、多クラスは(n_classes, n_features) の形状になるはずです。
        # 二値の場合は(1, n_features)なので .ravel() で1次元にできます。
        # 多クラスの場合はクラスごとの係数になるため、表示方法を検討する必要があります。
        # ここでは簡単のため、二値分類または multi_class='ovr' の場合を想定して、
        # 係数を1次元として扱える場合のみ実装します。
        # 多クラス ('multinomial') の場合は NotImplementedError とします。
        # ユーザーが multi_class を明示的に指定しない場合、'auto' で通常は大丈夫です。

        if self.model.coef_.shape[0] > 1:
            # 多クラス分類で、coef_がクラスごとの形状になっている場合
            # 例: (n_classes, n_features) -> multi_class='multinomial' または 'auto'で判別結果multinomial
             # この表示形式（特徴量ごとの単一の棒グラフ）には向かないため、一旦エラーとします。
             # 将来的にクラスごとの係数をHeatmapなどで表示する機能を検討してください。
             raise NotImplementedError("Multi-class Logistic Regression coefficients (coef_.shape[0] > 1) display is not yet supported in this format.")

        # 二値分類、または multi_class='ovr' の場合 (coef_.shape[0] == 1)
        coefficients = self.model.coef_[0] # 1行目の係数を取得し1次元配列にする

        # 切片 (Intercept) は特徴量重要度とは少し意味合いが違うため、DataFrameには含めないか別途表示
        # intercept = self.model.intercept_[0]


        # 特徴量名と係数をDataFrameにまとめる
        coefficients_df = pd.DataFrame({
            "feature": self.features,
            "coefficient": coefficients
        }).sort_values(by="coefficient", ascending=False) # 係数の値で降順にソート (正負あり)

        # 係数には正負があるため、絶対値でソートしたり、別途表示方法を検討することもできます。
        # ここでは値そのものでソートしておきます。

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
        # ロジスティック回帰は分類専用
        scoring_dict: dict[str, str] = {
            LogiRegMetric.ACCURACY.value: 'accuracy',
            LogiRegMetric.LOGLOSS.value: 'neg_log_loss' # 損失は負の値で返る
            # 多クラス分類の場合、roc_auc, precision, recall, f1 は average='weighted' などが必要
            # cross_val_score で単純な文字列指定が難しい場合があります。
            # make_scorer を使うか、ここでは単純な指標のみに限定します。
        }

        # 二値分類の場合のみ単純な roc_auc, precision, recall, f1 を追加
        # y_nunique() は CV の分割ごとに変わる可能性があるが、全体のyで判定
        if y.nunique() <= 2:
             scoring_dict[LogiRegMetric.AUC.value] = 'roc_auc'
             scoring_dict[LogiRegMetric.PRECISION.value] = 'precision'
             scoring_dict[LogiRegMetric.RECALL.value] = 'recall'
             scoring_dict[LogiRegMetric.F1.value] = 'f1'
        else:
            # 多クラス分類の場合のAUC (ovr)
            scoring_dict[LogiRegMetric.AUC.value] = 'roc_auc_ovr'


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

                # 回帰用 metrics はこのパイプラインでは計算しないため、RMSEの計算は不要

            except Exception as e:
                print(f"Warning: Could not compute cross-validation score for metric '{metric_name}' using scoring '{scoring_method}'. Error: {str(e)}")
                # エラーが発生したメトリックの結果はNaNなどで埋める
                results[metric_name] = np.full(cv, np.nan) # エラー時はNaNの配列で埋める


        # 結果をDataFrame形式で返す（各列が評価指標、各行がCVの分割結果）
        return pd.DataFrame(results)