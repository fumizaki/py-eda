import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, Literal


def draw(
    df: pd.DataFrame,
    x_col: str,
    hue: Optional[str] = None,
    orient: Optional[Literal["v", "h"]] = "v",
    palette: Optional[str] = "viridis",
    title: Optional[str] = None,
    height: Optional[int] = 15,
    width: Optional[int] = 30
) -> None:
    """
    Draw a count plot showing the counts of observations in each categorical bin.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): The column to count.
        hue (Optional[str]): Column to group by. Defaults to None.
        orient (Optional[Literal["v", "h"]]): Orientation: "v" (vertical) or "h" (horizontal). Defaults to "v".
        palette (Optional[str]): Color palette. Defaults to "viridis".
        title (Optional[str]): Title of the plot. If None, generates a title.
        height (Optional[int]): Height of the plot in pixels. Defaults to 15.
        width (Optional[int]): Width of the plot in pixels. Defaults to 30.
    """
    # Set context and style
    sns.set_context('poster')
    sns.set_style('whitegrid')
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Draw count plot
    if orient == "v":
        sns.countplot(
            data=df,
            x=x_col,
            # hue=hue,
            # palette=palette,
            ax=ax
        )
    else:  # horizontal
        sns.countplot(
            data=df,
            y=x_col,
            # hue=hue,
            # palette=palette,
            ax=ax
        )
    
    # Set title
    if title is None:
        title = f"Count of {x_col}"
        if hue:
            title += f" (grouped by {hue})"
    ax.set_title(title, fontsize=16)
    
    # Rotate x-axis labels if vertical and many categories
    if orient == "v" and len(df[x_col].unique()) > 5:
        plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)