import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional


def draw(
    df: pd.DataFrame,
    col: str,
    bins: Optional[int] = 20,
    kde: Optional[bool] = True,
    color: Optional[str] = "steelblue",
    title: Optional[str] = None,
    height: Optional[int] = 15,
    width: Optional[int] = 30
) -> None:
    """
    Draw a histogram of the specified column in the dataframe.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to plot.
        bins (Optional[int]): Number of bins for the histogram. Defaults to 20.
        kde (Optional[bool]): Whether to show kernel density estimate. Defaults to True.
        color (Optional[str]): Color of the histogram. Defaults to "steelblue".
        title (Optional[str]): Title of the plot. If None, uses column name.
        height (Optional[int]): Height of the plot in pixels. Defaults to 15.
        width (Optional[int]): Width of the plot in pixels. Defaults to 30.
    """
    # Set context and style
    sns.set_context('poster')
    sns.set_style('whitegrid')
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Draw histogram
    sns.histplot(
        data=df,
        x=col,
        bins=bins,
        kde=kde,
        color=color,
        ax=ax
    )
    
    # Set title
    if title is None:
        title = f"Distribution of {col}"
    ax.set_title(title, fontsize=16)
    
    # Set labels
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)