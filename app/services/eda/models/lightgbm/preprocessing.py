import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import List, Dict, Any, Tuple, Optional

class LightGBMPreprocessng:
    """
    LightGBMモデル学習のためのデータ前処理クラス。
    """
    def __init__(self, task_type: str):
        self.task_type = task_type # 'classification' or 'regression'
        self.preprocessing_steps: List[Dict[str, Any]] = []
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._one_hot_encoders: Dict[str, OneHotEncoder] = {}
        self._scalers: Dict[str, Any] = {}
        self._imputers: Dict[str, Any] = {}
        self.feature_names: List[str] = [] # 処理後の特徴量名

    def add_step(self, step_type: str, columns: List[str], params: Optional[Dict[str, Any]] = None):
        """
        前処理ステップを追加します。

        Args:
            step_type (str): ステップのタイプ (enum.PreprocessingStep の値を想定)
            columns (List[str]): 対象カラム名リスト
            params (Optional[Dict[str, Any]]): ステップ固有のパラメータ
        """
        if params is None:
            params = {}
        self.preprocessing_steps.append({
            'type': step_type,
            'columns': columns,
            'params': params
        })

    def apply(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        追加された前処理ステップをDataFrameに適用します。

        Args:
            df (pd.DataFrame): 処理対象のDataFrame。
            is_training (bool): 学習データに対して適用するかどうか。
                               Trueの場合、fitとtransformを行います。
                               Falseの場合、transformのみ行います。

        Returns:
            pd.DataFrame: 前処理後のDataFrame。
        """
        processed_df = df.copy()

        for step in self.preprocessing_steps:
            step_type = step['type']
            columns = step['columns']
            params = step['params']

            # 欠損値処理
            if step_type in ['Fill NA (Mean)', 'Fill NA (Median)', 'Fill NA (Mode)', 'Fill NA (Constant)']:
                strategy = step_type.split(' ')[2].lower() # 'mean', 'median', 'mode', 'constant'
                fill_value = params.get('fill_value') if strategy == 'constant' else None
                imputer_key = f"{step_type}_{'_'.join(columns)}"

                if is_training:
                    if strategy == 'constant' and fill_value is None:
                         raise ValueError("Constant fill requires 'fill_value' parameter.")
                    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
                    processed_df[columns] = imputer.fit_transform(processed_df[columns])
                    self._imputers[imputer_key] = imputer
                elif imputer_key in self._imputers:
                    processed_df[columns] = self._imputers[imputer_key].transform(processed_df[columns])
                else:
                     print(f"Warning: Imputer for {imputer_key} not found. Skipping imputation.")


            # エンコーディング
            elif step_type == 'Label Encoding':
                for col in columns:
                    encoder_key = f"label_{col}"
                    if is_training:
                        # NaNを考慮してfitする前にNaNを一時的に特定のplaceholderに置換
                        placeholder = '__NAN__'
                        original_nan_mask = processed_df[col].isnull()
                        processed_df[col] = processed_df[col].fillna(placeholder)

                        le = LabelEncoder()
                        processed_df[col] = le.fit_transform(processed_df[col])
                        self._label_encoders[encoder_key] = le

                        # 元のNaN位置に戻す (LabelEncoderは数値に変換するためNaNを保持できない)
                        # LightGBMはカテゴリー特徴量を扱えるため、後でLightGBMにカテゴリとして認識させるか、
                        # ここでNaNを特定の数値(-1など)に置き換える必要がある。
                        # シンプルにNaNを-1として扱う例：
                        processed_df.loc[original_nan_mask, col] = -1 # または適切な欠損値表現

                    elif encoder_key in self._label_encoders:
                        le = self._label_encoders[encoder_key]
                        # transformする前に未知の値やNaNを処理
                        # 未知の値はLabelEncoderのfit_transformではエラーになるため、事前に処理が必要
                        # 例：未知の値を-1にマッピング
                        processed_df[col] = processed_df[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
                        le.classes_ = np.append(le.classes_, '<unknown>') # 未知の値をクラスに追加
                        processed_df[col] = le.transform(processed_df[col])

                        # NaNの処理 (学習時と同じ値を適用)
                        processed_df[col] = processed_df[col].fillna(-1) # 学習時と同じ値

                    else:
                        print(f"Warning: LabelEncoder for {col} not found. Skipping encoding.")

            elif step_type == 'One-Hot Encoding':
                 encoder_key = f"onehot_{'_'.join(columns)}"
                 if is_training:
                     ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse=False for dense array
                     # 選択されたカラムのみをOne-Hot Encoding
                     encoded_data = ohe.fit_transform(processed_df[columns])
                     # 元のDataFrameから選択されたカラムを削除し、エンコードされたデータを結合
                     processed_df = processed_df.drop(columns=columns)
                     encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(columns), index=processed_df.index)
                     processed_df = pd.concat([processed_df, encoded_df], axis=1)
                     self._one_hot_encoders[encoder_key] = ohe
                 elif encoder_key in self._one_hot_encoders:
                     ohe = self._one_hot_encoders[encoder_key]
                     # transformする前に未知の値やNaNを処理
                     # transform時に未知の値が含まれる場合はhandle_unknown='ignore'で0ベクトルになる
                     encoded_data = ohe.transform(processed_df[columns])
                     processed_df = processed_df.drop(columns=columns)
                     encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(columns), index=processed_df.index)
                     processed_df = pd.concat([processed_df, encoded_df], axis=1)
                 else:
                     print(f"Warning: OneHotEncoder for {columns} not found. Skipping encoding.")

            # スケーリング
            elif step_type == 'Min-Max Scaling':
                 scaler_key = f"minmax_{'_'.join(columns)}"
                 if is_training:
                     scaler = MinMaxScaler()
                     # NaNを考慮しない場合はdropna().valuesを使うか、impute後にスケーリングする
                     # ここではシンプルにimpute後を想定
                     processed_df[columns] = scaler.fit_transform(processed_df[columns])
                     self._scalers[scaler_key] = scaler
                 elif scaler_key in self._scalers:
                     scaler = self._scalers[scaler_key]
                     processed_df[columns] = scaler.transform(processed_df[columns])
                 else:
                     print(f"Warning: MinMaxScaler for {columns} not found. Skipping scaling.")

            elif step_type == 'Standard Scaling':
                scaler_key = f"standard_{'_'.join(columns)}"
                if is_training:
                     scaler = StandardScaler()
                     processed_df[columns] = scaler.fit_transform(processed_df[columns])
                     self._scalers[scaler_key] = scaler
                elif scaler_key in self._scalers:
                     scaler = self._scalers[scaler_key]
                     processed_df[columns] = scaler.transform(processed_df[columns])
                else:
                     print(f"Warning: StandardScaler for {columns} not found. Skipping scaling.")

            # カラム削除
            elif step_type == 'Drop Columns':
                 processed_df = processed_df.drop(columns=columns, errors='ignore')


        # 処理後の特徴量名を更新
        self.feature_names = processed_df.columns.tolist()

        return processed_df

    def split_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        データを学習用とテスト用に分割します。

        Args:
            df (pd.DataFrame): 分割対象のDataFrame。
            target_column (str): 目的変数カラム名。
            test_size (float): テストデータの割合。
            random_state (int): シャッフルや分割のための乱数シード。

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if self.task_type == 'classification' else None)

        return X_train, X_test, y_train, y_test