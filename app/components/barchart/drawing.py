import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, Literal


def draw_barchart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue: Optional[str] = None,
    orient: Optional[Literal["v", "h"]] = "v",
    palette: Optional[str] = "viridis",
    title: Optional[str] = None,
    height: Optional[int] = 15,
    width: Optional[int] = 30
) -> None:
    """
    Draw a bar chart of the specified columns in the dataframe.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): The column for the x-axis (or y-axis if orient="h").
        y_col (str): The column for the y-axis (or x-axis if orient="h").
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
    
    # Determine x and y based on orientation
    x, y = (x_col, y_col) if orient == "v" else (y_col, x_col)
    
    # Draw bar chart
    sns.barplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        ax=ax
    )
    
    # Set title
    if title is None:
        title = f"{y_col} by {x_col}"
        if hue:
            title += f" (grouped by {hue})"
    ax.set_title(title, fontsize=16)
    
    # Rotate x-axis labels if vertical and many categories
    if orient == "v" and len(df[x].unique()) > 5:
        plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)