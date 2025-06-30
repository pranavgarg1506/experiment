import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from langchain.tools import tool
import os


def read_data() -> pd.DataFrame:
    """Reads the dataset from a CSV file."""
    if not os.path.exists("data/trades.csv"):
        raise FileNotFoundError("Dataset not found. Please ensure 'data/trades.csv' exists.")
    return pd.read_csv("data/trades.csv")

def perform_eda(df: pd.DataFrame) -> str:
    print(type(df),df)
    desc = df.describe().to_string()
    missing = df.isnull().sum()
    return f"Descriptive Stats:\n{desc}\n\nMissing Values:\n{missing.to_string()}"

def feature_selection(df: pd.DataFrame, target_column: str, k: int = 5) -> list:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_cols = X.columns[selector.get_support()].tolist()
    return selected_cols

def correlation_heatmap(df: pd.DataFrame, path="heatmap.png") -> str:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(path)
    return f"Correlation heatmap saved to {path}"

@tool
def run_eda(df) -> str:
    """Performs EDA on the dataset."""
    return perform_eda(df)

@tool
def run_feature_selection(df) -> str:
    """Selects top features based on ANOVA F-score."""
    top_features = feature_selection(df, target_column='target', k=5)
    return f"Top features: {', '.join(top_features)}"

@tool
def run_heatmap(df) -> str:
    """Generates a correlation heatmap from the dataset."""
    return correlation_heatmap(df)


@tool
def run_read_data() -> pd.DataFrame:
    """Reads the dataset from a CSV file."""
    return read_data()