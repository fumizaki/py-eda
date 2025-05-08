import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import PolynomialFeatures

class FeatureEngineer:
    """
    LightGBMモデルのための特徴量エンジニアリングクラス。
    """
    def __init__(self):
        self.engineering_steps: List[Dict[str, Any]] = []
        self._polynomial_transformers: Dict[str, PolynomialFeatures] = {} # ポリノミアル特徴量用

    def add_step(self, step_type: str, columns: List[str], params: Optional[Dict[str, Any]] = None):
        """
        特徴量エンジニアリングステップを追加します。

        Args:
            step_type (str): ステップのタイプ (enum.FeatureEngineeringStep の値を想定)
            columns (List[str]): 対象カラム名リスト
            params (Optional[Dict[str, Any]]): ステップ固有のパラメータ
        """
        if params is None:
            params = {}
        self.engineering_steps.append({
            'type': step_type,
            'columns': columns,
            'params': params
        })

    def apply(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        追加された特徴量エンジニアリングステップをDataFrameに適用します。

        Args:
            df (pd.DataFrame): 処理対象のDataFrame。
            is_training (bool): 学習データに対して適用するかどうか。
                               Trueの場合、fitとtransformを行います。
                               Falseの場合、transformのみ行います。

        Returns:
            pd.DataFrame: 特徴量エンジニアリング後のDataFrame。
        """
        engineered_df = df.copy()

        for step in self.engineering_steps:
            step_type = step['type']
            columns = step['columns']
            params = step['params']

            if step_type == 'Datetime Features':
                for col in columns:
                    if pd.api.types.is_datetime64_any_dtype(engineered_df[col]):
                        try:
                            # 時系列特徴量を生成
                            engineered_df[f'{col}_year'] = engineered_df[col].dt.year
                            engineered_df[f'{col}_month'] = engineered_df[col].dt.month
                            engineered_df[f'{col}_day'] = engineered_df[col].dt.day
                            engineered_df[f'{col}_dayofweek'] = engineered_df[col].dt.dayofweek
                            engineered_df[f'{col}_hour'] = engineered_df[col].dt.hour
                            engineered_df[f'{col}_minute'] = engineered_df[col].dt.minute
                            engineered_df[f'{col}_quarter'] = engineered_df[col].dt.quarter
                            engineered_df[f'{col}_is_month_start'] = engineered_df[col].dt.is_month_start.astype(int)
                            engineered_df[f'{col}_is_month_end'] = engineered_df[col].dt.is_month_end.astype(int)
                            engineered_df[f'{col}_is_year_start'] = engineered_df[col].dt.is_year_start.astype(int)
                            engineered_df[f'{col}_is_year_end'] = engineered_df[col].dt.is_year_end.astype(int)
                            engineered_df[f'{col}_weekofyear'] = engineered_df[col].dt.isocalendar().week.astype(int)
                        except Exception as e:
                            print(f"Warning: Could not create datetime features for column '{col}': {e}")
                    else:
                        print(f"Warning: Column '{col}' is not a datetime type. Skipping datetime feature engineering.")

            elif step_type == 'Polynomial Features':
                degree = params.get('degree', 2)
                interaction_only = params.get('interaction_only', False)
                poly_key = f"poly_{'_'.join(columns)}_{degree}_{interaction_only}"

                try:
                    # NaNを含む場合は事前に処理が必要。ここではNaNなしを前提とする。
                    if is_training:
                        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
                        poly_features = poly.fit_transform(engineered_df[columns])
                        self._polynomial_transformers[poly_key] = poly
                    elif poly_key in self._polynomial_transformers:
                        poly = self._polynomial_transformers[poly_key]
                        poly_features = poly.transform(engineered_df[columns])
                    else:
                         print(f"Warning: PolynomialFeatures transformer for {columns} not found. Skipping transformation.")
                         continue # スキップして次のステップへ

                    # 新しい特徴量名を生成してDataFrameに結合
                    feature_names = poly.get_feature_names_out(columns)
                    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=engineered_df.index)
                    engineered_df = pd.concat([engineered_df, poly_df], axis=1)

                except Exception as e:
                     print(f"Warning: Could not create polynomial features for columns '{columns}': {e}")


            elif step_type == 'Interaction Features':
                # 例: 'col1', 'col2' -> 'col1_x_col2'
                for i in range(len(columns)):
                    for j in range(i + 1, len(columns)):
                        col1 = columns[i]
                        col2 = columns[j]
                        interaction_col_name = f'{col1}_x_{col2}'
                        if col1 in engineered_df.columns and col2 in engineered_df.columns:
                            # 数値型カラムのみを対象とする
                            if pd.api.types.is_numeric_dtype(engineered_df[col1]) and pd.api.types.is_numeric_dtype(engineered_df[col2]):
                                engineered_df[interaction_col_name] = engineered_df[col1] * engineered_df[col2]
                            else:
                                print(f"Warning: Interaction features skipped for non-numeric columns '{col1}' and '{col2}'.")
                        else:
                            print(f"Warning: Columns '{col1}' or '{col2}' not found for interaction feature.")


            elif step_type == 'Group By Features':
                 # 例: カテゴリ変数 'category_col' で数値変数 'value_col' を集約
                 # params: {'by': 'category_col', 'agg_column': 'value_col', 'agg_funcs': ['mean', 'sum']}
                 group_by_col = params.get('by')
                 agg_column = params.get('agg_column')
                 agg_funcs = params.get('agg_funcs') # e.g., ['mean', 'sum', 'count', 'max', 'min']

                 if group_by_col and agg_column and agg_funcs:
                     if group_by_col in engineered_df.columns and agg_column in engineered_df.columns:
                         try:
                             # グループごとの集計値を計算
                             agg_features = engineered_df.groupby(group_by_col)[agg_column].agg(agg_funcs)

                             # 元のDataFrameに結合 (マージ)
                             # インデックスが一致していることを確認
                             if engineered_df.index.equals(agg_features.index):
                                 # 集計結果の新しいカラム名を生成
                                 new_col_names = [f'{agg_column}_by_{group_by_col}_{func}' for func in agg_funcs]
                                 agg_features.columns = new_col_names
                                 engineered_df = pd.concat([engineered_df, agg_features], axis=1)
                             else:
                                # グループ化のaggregationは通常元の行数と同じにならないため、マージが必要
                                 agg_features.columns = [f'{agg_column}_by_{group_by_col}_{func}' for func in agg_funcs]
                                 engineered_df = pd.merge(engineered_df, agg_features, left_on=group_by_col, right_index=True, how='left')

                         except Exception as e:
                              print(f"Warning: Could not create group by features for '{agg_column}' by '{group_by_col}': {e}")
                     else:
                         print(f"Warning: Group by column '{group_by_col}' or aggregate column '{agg_column}' not found.")
                 else:
                      print("Warning: Missing 'by', 'agg_column', or 'agg_funcs' parameters for Group By Features.")

        return engineered_df