import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional

def diagnoal_kind_options() -> list[str]:
    return [
        "kde",
        "hist"
    ]

def draw(
    df: pd.DataFrame,
    columns: list[str],
    hue: Optional[str] = None,
    diag_kind: Optional[str] = "kde",
    palette: Optional[str] = "tab10",
    title: Optional[str] = None,
    height: Optional[float] = 2.5
) -> None:
    """
    Draw a pairplot (scatterplot matrix) of the specified columns in the dataframe.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (List[str]): List of columns to include in the pairplot.
        hue (Optional[str]): Column to color-code points by. Defaults to None.
        diag_kind (Optional[str]): Kind of plot for diagonal: "kde" or "hist". Defaults to "kde".
        palette (Optional[str]): Color palette. Defaults to "tab10".
        height (Optional[float]): Height of each facet in inches. Defaults to 2.5.
    """
    # Set context and style
    sns.set_context('talk')
    
    # Draw pairplot
    pairplot = sns.pairplot(
        data=df,
        vars=columns,
        hue=hue,
        diag_kind=diag_kind,
        # palette=palette,
        height=height
    )
    
    # Set title
    if title is None:
        title = f"Pair Plot"
    plt.title(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(pairplot.figure)