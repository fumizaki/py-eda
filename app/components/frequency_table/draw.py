import streamlit as st
import pandas as pd
from typing import Optional


def draw(df: pd.DataFrame, col: str, title: Optional[str] = None):
    """
    単一のカテゴリカル変数の度数分布表をstreamlitのdataframeとして表示します。

    Args:
        df (pd.DataFrame): 描画するデータフレーム。
        col (str): 度数分布表を表示するカテゴリカルカラムの名前。
        title (Optional[str]): Title of the plot. If None, generates a title.
    """

    frequency_counts = df[col].value_counts().sort_index()
    percentage = df[col].value_counts(normalize=True).sort_index().mul(100).round(2)
    cumulative_frequency = frequency_counts.cumsum()
    cumulative_percentage = percentage.cumsum().round(2)

    frequency_df = pd.DataFrame({
        'カテゴリ': frequency_counts.index,
        '度数': frequency_counts.values,
        '相対度数 (%)': percentage.values,
        '累積度数': cumulative_frequency.values,
        '累積相対度数 (%)': cumulative_percentage.values
    })

    # Set title
    if title is None:
        title = f"{col}の度数と割合"

    st.subheader(title)
    st.dataframe(frequency_df, hide_index=True)