from typing import IO, Optional
import pandas as pd
import numpy as np
from services.dataset.dataset_repository import DatasetRepository
from .enum import CorrelationMethod

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


    def stats(self, include='all'):
        self.check_df_loaded()
        return self.df.describe(include=include)

    
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
        high_corr_pairs.columns = ['variable1', 'variable2', 'correlation']

        # 絶対値でソート
        if not high_corr_pairs.empty:
            return high_corr_pairs.sort_values('correlation', key=abs, ascending=False)

        else:
            return pd.DataFrame(columns=['variable1', 'variable2', 'correlation'])

