import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, Literal

def inner_options() -> list[str]:
    return [
        "box",
        "quart",
        "point",
        "stick",
    ]

def draw(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    hue: Optional[str] = None,
    split: Optional[bool] = False,
    inner: Optional[Literal["box", "quart", "point", "stick", None]] = "box",
    palette: Optional[str] = "muted",
    title: Optional[str] = None,
    height: Optional[int] = 15,
    width: Optional[int] = 30
) -> None:
    """
    Draw a violin plot to visualize distribution of data and its probability density.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        y_col (str): The column for the y-axis.
        x_col (Optional[str]): The column for the x-axis. Defaults to None.
        hue (Optional[str]): Column to group by. Defaults to None.
        split (Optional[bool]): If True, and hue is used, draw half violins. Defaults to False.
        inner (Optional[Literal]): Representation inside violin. Defaults to "box".
        palette (Optional[str]): Color palette. Defaults to "muted".
        title (Optional[str]): Title of the plot. If None, generates a title.
        height (Optional[int]): Height of the plot in pixels. Defaults to 15.
        width (Optional[int]): Width of the plot in pixels. Defaults to 30.
    """
    # Set context and style
    sns.set_context('poster')
    sns.set_style('whitegrid')
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Draw violin plot
    sns.violinplot(
        data=df,
        x=x_col,
        y=y_col,
        # hue=hue,
        split=split,
        inner=inner,
        # palette=palette,
        ax=ax
    )
    
    # Set title
    if title is None:
        title = f"Distribution of {y_col}"
        if x_col:
            title += f" by {x_col}"
    ax.set_title(title, fontsize=16)
    
    # Rotate x-axis labels if many categories
    if x_col and len(df[x_col].unique()) > 5:
        plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)