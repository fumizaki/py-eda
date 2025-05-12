from typing import IO, Optional, Union
import pandas as pd
import numpy as np
from services.dataset.dataset_repository import DatasetRepository
from .enum import DescribeType, CorrelationMethod, ImputeMethod, DetectOutlierMethod, TreatOutlierMethod, ScalingMethod

class EDADataFrame:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.dataset_repository: DatasetRepository = DatasetRepository('datasets')
        self.columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.other_columns: list[str] = []
        self.missing_columns: list[str] = []
        self.unique_columns: list[str] = []
        self.duplicated_columns: list[str] = []


    def check_df_loaded(self):
        if self.df is None:
            raise ValueError("DataFrame is not loaded yet.")


    def load(self, df: pd.DataFrame) -> None:
        self.df = df
        self.update_column()


    def load_from_csv(self, csv: IO[bytes]) -> None:
        self.load(pd.read_csv(csv))


    def load_from_option(self, option: str) -> None:
        self.load(self.dataset_repository.get_dataframe(option))


    def dataset_options(self) -> list[str]:
        return self.dataset_repository.options


    def find_missing_columns(self) -> pd.Series:
        self.check_df_loaded()
        return self.df.isnull().sum()[self.df.isnull().any()]


    def update_column(self) -> None:
        self.check_df_loaded()

        self.columns = self.df.columns.tolist()
        self.numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.other_columns = [col for col in self.columns if col not in self.numeric_columns and col not in self.categorical_columns]
        self.missing_columns = self.find_missing_columns().index.tolist()
        self.unique_columns = [col for col in self.df.columns if self.df[col].nunique() == self.df[col].count()]
        self.duplicated_columns = [col for col in self.df.columns if self.df[col].dropna().duplicated().any()]


    def stats(self, include: DescribeType = DescribeType.ALL):
        self.check_df_loaded()
        if include == DescribeType.ALL:
            return self.df.describe(include = DescribeType.ALL.value)
        elif include == DescribeType.NUMBER:
            return self.df.describe(include = np.number)
        elif include == DescribeType.OBJECT:
             return self.df.describe(include = [DescribeType.OBJECT.value, DescribeType.CATEGORY.value])
        else:
             return self.df.describe(include = DescribeType.ALL.value)

    
    def get_correlation_matrix(self, columns: list[str], method: CorrelationMethod = CorrelationMethod.PEARSON) -> pd.DataFrame:
        self.check_df_loaded()

        if len(columns) < 2:
            raise ValueError("相関分析には少なくとも2つの数値型カラムが必要です。")

        # 相関係数行列を計算
        corr_matrix = self.df[columns].corr(method=method)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        return corr_matrix.where(mask).T


    def get_high_correlations(self, columns: list[str], method: CorrelationMethod = CorrelationMethod.PEARSON, threshold: float = 0.7) -> pd.DataFrame:
        self.check_df_loaded()

        # 相関行列を取得
        corr_matrix = self.get_correlation_matrix(columns, method).abs()

        # 上三角行列をマスク
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # マスクを適用して閾値以上の要素を持つDataFrameを作成
        upper_triangle = corr_matrix.mask(mask)
        high_corr_pairs = upper_triangle[upper_triangle >= threshold].stack().reset_index()
        high_corr_pairs.columns = ['変数1', '変数2', '相関']

        # 絶対値でソート
        if not high_corr_pairs.empty:
            return high_corr_pairs.sort_values('相関', key=abs, ascending=False)

        else:
            return pd.DataFrame(columns=['変数1', '変数2', '相関'])


    def impute_missing_value(self, column: str, value: Optional[ str | int | float ] = None, method: Optional[ImputeMethod] = None, groupby: Optional[list[str]] = None) -> None:
        self.check_df_loaded()

        if column not in self.columns:
            raise ValueError(f"指定されたカラム '{column}' は存在しません。")
        
        if value is None and method is None:
            raise ValueError("補完には 'value' または 'method' のいずれかを指定する必要があります。")

        col_series = self.df[column]

        # 欠損値がない場合は何もしない
        if col_series.isnull().sum() == 0:
            print(f"カラム '{column}' には欠損値がありません。補完はスキップされました。")
            return
        
        
        if value:
            self.df[column] = col_series.fillna(value)

        else:
            if groupby: # groupby が指定されている場合
                # groupby カラムの存在チェック
                if not all(g_col in self.columns for g_col in groupby):
                    missing_groupby_cols = [g_col for g_col in groupby if g_col not in self.columns]
                    raise ValueError(f"指定されたgroupbyカラム {missing_groupby_cols} が存在しません。")

                # groupby 処理が可能なメソッドかチェック
                if method not in [ImputeMethod.MEAN, ImputeMethod.MEDIAN, ImputeMethod.MODE]:
                    raise ValueError(f"groupby が指定されていますが、メソッド '{method.value}' は groupby 処理をサポートしていません。")

                else: # method は MEAN, MEDIAN, MODE のいずれか (groupby処理を実行)
                    # 数値型チェック (MEAN, MEDIAN の場合)
                    if method in [ImputeMethod.MEAN, ImputeMethod.MEDIAN] and column not in self.numeric_columns:
                        raise ValueError(f"'{method.value}' メソッド (groupbyあり) は数値型カラム '{column}' にのみ適用可能です（補完対象カラム）。")

                    # groupby 処理の実行
                    if method == ImputeMethod.MEAN:
                        filled_series = self.df.groupby(groupby)[column].transform(ImputeMethod.MEAN.value)
                    elif method == ImputeMethod.MEDIAN:
                        filled_series = self.df.groupby(groupby)[column].transform(ImputeMethod.MEDIAN.value)
                    elif method == ImputeMethod.MODE:
                        # groupby modeの場合はカスタム関数が必要
                        # transformは単一の値を返す必要があるため、mode()[0]を使用
                        # グループが空または全て欠損値の場合はmode()が空になるため、np.nanを返すように処理
                        try:
                            filled_series = self.df.groupby(groupby)[column].transform(lambda x: x.mode()[0] if not pd.Series(x).mode().empty else np.nan)
                        except Exception as e:
                            print(f"groupby_mode 処理中にエラーが発生しました (カラム: {column}, グループ: {groupby}): {e}. 一部のグループの欠損値が完全に補完されない可能性があります。")
                            # エラーが発生した場合でも、部分的に補完された結果を適用
                            filled_series = self.df.groupby(groupby)[column].transform(lambda x: x.mode()[0] if not pd.Series(x).mode().empty else np.nan) # 再度試行

                    # 補完の実行
                    self.df[column] = col_series.fillna(filled_series)

            # groupby が None の場合
            if groupby is None or method not in [ImputeMethod.MEAN, ImputeMethod.MEDIAN, ImputeMethod.MODE]:
                if method == ImputeMethod.MEAN:
                    if column not in self.numeric_columns:
                        raise ValueError(f"'{method.value}' メソッドは数値型カラム '{column}' にのみ適用可能です。")
                    fill_value_calc = col_series.mean()
                    self.df[column] = col_series.fillna(fill_value_calc)

                elif method == ImputeMethod.MEDIAN:
                    if column not in self.numeric_columns:
                        raise ValueError(f"'{method.value}' メソッドは数値型カラム '{column}' にのみ適用可能です。")
                    fill_value_calc = col_series.median()
                    self.df[column] = col_series.fillna(fill_value_calc)

                elif method == ImputeMethod.MODE:
                    # mode()は複数のモードを返す可能性があるため、最初の要素を選択
                    fill_values_calc = col_series.mode()
                    if not fill_values_calc.empty:
                        # 最頻値が複数ある場合は最初の要素を使用
                        fill_value_calc = fill_values_calc[0]
                        self.df[column] = col_series.fillna(fill_value_calc)
                    else:
                        print(f"カラム '{column}' に最頻値が計算できませんでした（データが全て欠損値など）。欠損値は補完されません。")

                elif method == ImputeMethod.FFILL:
                    # Forward Fill。時系列データなどで有効。
                    self.df[column] = col_series.fillna(method=ImputeMethod.FFILL.value)

                elif method == ImputeMethod.BFILL:
                    # Backward Fill。時系列データなどで有効。
                    self.df[column] = col_series.fillna(method=ImputeMethod.BFILL.value)

                else:
                    # ここには到達しないはずだが念のため (method is not None なので)
                    raise ValueError(f"予期しない補完方法が指定されました: {method}")



        # 補完後にDataFrameが変更されたので、カラム情報を更新
        self.update_column()

            
    def handle_outlier(
        self,
        column: str,
        detection_method: DetectOutlierMethod,
        threshold: Union[float, tuple[float, float]],
        treatment_method: TreatOutlierMethod,
        replace_value: Optional[str | int | float | bool] = None
    ) -> None:
        """
        Handle outliers in a specified column using various detection and treatment methods.
        
        Args:
            column (str): The name of the column to handle outliers in.
            detection_method (DetectOutlierMethod): Method to detect outliers.
            threshold (Union[float, tuple[float, float]]): 
                - For IQR: Multiplier for IQR (typically 1.5)
                - For Z-Score: Number of standard deviations
                - For Percentile: Tuple of (lower_percentile, upper_percentile)
            treatment_method (TreatOutlierMethod): Method to treat detected outliers.
            replace_value (Optional): Value to replace outliers with if treatment method is REPLACE.
        
        Raises:
            ValueError: If the column is invalid or the method is not applicable.
        """
        self.check_df_loaded()

        # Validate column existence
        if column not in self.columns:
            raise ValueError(f"Specified column '{column}' does not exist.")
        
        # Validate column is numeric for most methods
        if column not in self.numeric_columns and detection_method != DetectOutlierMethod.PERCENTILE:
            raise ValueError(f"Outlier detection method '{detection_method.value}' requires a numeric column.")
        
        # Get the column series
        col_series = self.df[column]
        
        # Detect outliers based on method
        if detection_method == DetectOutlierMethod.IQR:
            # IQR method
            Q1 = col_series.quantile(0.25)
            Q3 = col_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            outlier_mask = (col_series < lower_bound) | (col_series > upper_bound)
        
        elif detection_method == DetectOutlierMethod.ZSCORE:
            # Z-Score method
            mean = col_series.mean()
            std = col_series.std()
            z_scores = np.abs((col_series - mean) / std)
            outlier_mask = z_scores > threshold
        
        elif detection_method == DetectOutlierMethod.PERCENTILE:
            # Percentile method (works with both numeric and non-numeric columns)
            if not isinstance(threshold, tuple) or len(threshold) != 2:
                raise ValueError("Percentile method requires a tuple of (lower_percentile, upper_percentile)")
            lower_percentile, upper_percentile = threshold
            lower_bound = col_series.quantile(lower_percentile / 100)
            upper_bound = col_series.quantile(upper_percentile / 100)
            outlier_mask = (col_series < lower_bound) | (col_series > upper_bound)
        
        else:
            raise ValueError(f"Unsupported detection method: {detection_method}")
        
        # Treat outliers based on method
        if treatment_method == TreatOutlierMethod.REMOVE:
            # Remove outlier rows
            self.df = self.df[~outlier_mask]
        
        elif treatment_method == TreatOutlierMethod.CLIP:
            # Clip outliers to boundary values
            if detection_method == DetectOutlierMethod.IQR:
                self.df.loc[col_series < lower_bound, column] = lower_bound
                self.df.loc[col_series > upper_bound, column] = upper_bound
            elif detection_method == DetectOutlierMethod.ZSCORE:
                mean = col_series.mean()
                std = col_series.std()
                lower_bound = mean - (threshold * std)
                upper_bound = mean + (threshold * std)
                self.df.loc[col_series < lower_bound, column] = lower_bound
                self.df.loc[col_series > upper_bound, column] = upper_bound
            elif detection_method == DetectOutlierMethod.PERCENTILE:
                self.df.loc[col_series < lower_bound, column] = lower_bound
                self.df.loc[col_series > upper_bound, column] = upper_bound
        
        elif treatment_method == TreatOutlierMethod.REPLACE:
            # Replace outliers with specified value
            if replace_value is None:
                raise ValueError("Replace method requires a replace_value to be specified")
            
            if detection_method == DetectOutlierMethod.IQR:
                self.df.loc[(col_series < lower_bound) | (col_series > upper_bound), column] = replace_value
            elif detection_method == DetectOutlierMethod.ZSCORE:
                mean = col_series.mean()
                std = col_series.std()
                lower_bound = mean - (threshold * std)
                upper_bound = mean + (threshold * std)
                self.df.loc[(col_series < lower_bound) | (col_series > upper_bound), column] = replace_value
            elif detection_method == DetectOutlierMethod.PERCENTILE:
                self.df.loc[(col_series < lower_bound) | (col_series > upper_bound), column] = replace_value
        
        else:
            raise ValueError(f"Unsupported treatment method: {treatment_method}")
        
        # Update column information after handling outliers
        self.update_column()
        
        
        
    def scale_columns(
        self, 
        columns: list[str], 
        method: ScalingMethod = ScalingMethod.STANDARD,
        scale_range: Optional[tuple[float, float]] = None
    ) -> None:
        """
        Scale specified numeric columns using the chosen scaling method
        
        Args:
            columns (list[str]): List of columns to scale. 
            method (ScalingMethod): Scaling method to apply.
            scale_range (Optional[tuple[float, float]]): Custom range for MinMax scaling. 
                                                        Defaults to (0, 1) if not specified.
        
        Raises:
            ValueError: If no numeric columns are found or invalid inputs are provided.
        """
        # Validate DataFrame is loaded
        self.check_df_loaded()
        
        # Validate columns
        if not columns:
            raise ValueError("スケーリングする数値型カラムがありません。")
        
        # Validate all specified columns are numeric
        invalid_columns = [col for col in columns if col not in self.numeric_columns]
        if invalid_columns:
            raise ValueError(f"指定されたカラム {invalid_columns} は数値型ではありません。")
        
        # Perform scaling based on method
        if method == ScalingMethod.STANDARD:
            # Z-score normalization (zero mean, unit variance)
            for col in columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std == 0:
                    # Skip columns with zero standard deviation
                    print(f"警告: カラム '{col}' は定数値のためスケーリングをスキップします。")
                    continue
                self.df[col] = (self.df[col] - mean) / std
        
        elif method == ScalingMethod.MINMAX:
            # MinMax scaling
            if scale_range is None:
                scale_range = (0, 1)  # Default to [0, 1]
            
            for col in columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                
                # Check for constant column
                if min_val == max_val:
                    print(f"警告: カラム '{col}' は定数値のためスケーリングをスキップします。")
                    continue
                
                # Apply MinMax scaling to specified range
                self.df[col] = (self.df[col] - min_val) / (max_val - min_val) * (scale_range[1] - scale_range[0]) + scale_range[0]
        
        elif method == ScalingMethod.ROBUST:
            # Robust scaling using median and IQR
            for col in columns:
                median = self.df[col].median()
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Check for zero IQR
                if IQR == 0:
                    print(f"警告: カラム '{col}' は定数値のためスケーリングをスキップします。")
                    continue
                
                self.df[col] = (self.df[col] - median) / IQR
        
        elif method == ScalingMethod.MAXABS:
            # MaxAbs scaling
            for col in columns:
                max_abs_val = max(abs(self.df[col].min()), abs(self.df[col].max()))
                
                # Check for zero max absolute value
                if max_abs_val == 0:
                    print(f"警告: カラム '{col}' は定数値のためスケーリングをスキップします。")
                    continue
                
                self.df[col] = self.df[col] / max_abs_val
        
        else:
            raise ValueError(f"サポートされていないスケーリング方法: {method}")
        
        # Update column information
        self.update_column()