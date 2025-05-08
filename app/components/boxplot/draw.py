import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional


def draw(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    hue: Optional[str] = None,
    palette: Optional[str] = "Set2",
    title: Optional[str] = None,
    orient: Optional[str] = "v",
    height: Optional[int] = 15,
    width: Optional[int] = 30
) -> None:
    """
    Draw a boxplot of the specified column(s) in the dataframe.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        y_col (str): The column for the y-axis (or x-axis if orient="h").
        x_col (Optional[str]): The column for the x-axis (or y-axis if orient="h"). Defaults to None.
        hue (Optional[str]): Column to group by. Defaults to None.
        palette (Optional[str]): Color palette. Defaults to "Set2".
        title (Optional[str]): Title of the plot. If None, generates a title.
        orient (Optional[str]): Orientation: "v" (vertical) or "h" (horizontal). Defaults to "v".
        height (Optional[int]): Height of the plot in pixels. Defaults to 15.
        width (Optional[int]): Width of the plot in pixels. Defaults to 30.
    """
    # Set context and style
    sns.set_context('poster')
    sns.set_style('whitegrid')
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Determine x and y based on orientation
    x = x_col if orient == "v" else y_col
    y = y_col if orient == "v" else x_col
    
    # Draw boxplot
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        # hue=hue,
        # palette=palette,
        orient=orient,
        ax=ax
    )
    
    # Set title
    if title is None:
        if x_col:
            title = f"Distribution of {y_col} by {x_col}"
        else:
            title = f"Distribution of {y_col}"
    ax.set_title(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)