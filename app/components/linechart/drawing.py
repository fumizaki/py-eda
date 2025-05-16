import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional


def draw_linechart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue: Optional[str] = None,
    palette: Optional[str] = "viridis",
    title: Optional[str] = None,
    height: Optional[int] = 8,  # Adjust default height for line plots
    width: Optional[int] = 15   # Adjust default width for line plots
) -> None:
    """
    Draw a line chart of the specified columns in the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): The column for the x-axis.
        y_col (str): The column for the y-axis.
        hue (Optional[str]): Column to group by and color lines. Defaults to None.
        palette (Optional[str]): Color palette. Defaults to "viridis".
        title (Optional[str]): Title of the plot. If None, generates a title.
        height (Optional[int]): Height of the plot in inches. Defaults to 8.
        width (Optional[int]): Width of the plot in inches. Defaults to 15.
                          Note: figsize units are inches, not pixels.
    """
    # Set context and style
    sns.set_context('poster')
    sns.set_style('whitegrid')

    # Create figure and axes
    # Adjust figsize units are typically inches for matplotlib
    fig, ax = plt.subplots(figsize=(width, height))

    # Draw line chart
    # Ensure x_col data type is suitable for numerical or datetime x-axis
    # sns.lineplot can handle numerical, datetime, or categorical data on x,
    # but the interpretation changes. Numerical/datetime is most common for line plots.
    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue,
        palette=palette,
        ax=ax
    )

    # Set title
    if title is None:
        title = f"{y_col} over {x_col}"
        if hue:
            title += f" (grouped by {hue})"
    ax.set_title(title, fontsize=16)

    # Optional: Improve date formatting on x-axis if x_col is datetime
    # You might need to add specific logic here depending on the data type and range

    # Adjust layout
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free up memory
