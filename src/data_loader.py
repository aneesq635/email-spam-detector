"""Module for loading and normalizing email/SMS spam dataset."""

import os
import glob
from typing import Tuple, Optional
import pandas as pd


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Auto-detect the text and label columns from a dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        Tuple[str, str]: (text_column_name, label_column_name)

    Raises:
        ValueError: If suitable text or label columns cannot be determined.
    """
    cols_lower = {col: str(col).lower().strip() for col in df.columns}

    # Possible candidate names for text
    text_candidates = ["text", "message", "email", "content", "sms", "body", "v2"]
    # Possible candidate names for label
    label_candidates = ["label", "target", "category", "class", "v1", "label_num", "type"]

    detected_text_col: Optional[str] = None
    detected_label_col: Optional[str] = None

    # Check candidates
    for cand in text_candidates:
        for original_col, lower_col in cols_lower.items():
            if lower_col == cand:
                detected_text_col = original_col
                break
        if detected_text_col:
            break

    for cand in label_candidates:
        for original_col, lower_col in cols_lower.items():
            if lower_col == cand:
                detected_label_col = original_col
                break
        if detected_label_col:
            break

    # Fallback heuristic: look for object/string column with highest average length for text
    if not detected_text_col:
        text_cols = [c for c in df.columns if df[c].dtype == "object" and c != detected_label_col]
        if text_cols:
            detected_text_col = max(
                text_cols,
                key=lambda c: df[c].dropna().astype(str).str.len().mean() if len(df[c].dropna()) > 0 else 0
            )

    # Fallback heuristic: look for low-cardinality column for label (usually 2 unique values)
    if not detected_label_col:
        label_cols = [c for c in df.columns if c != detected_text_col and df[c].nunique() in [2, 3]]
        if label_cols:
            detected_label_col = label_cols[0]

    if not detected_text_col or not detected_label_col:
        raise ValueError(
            f"Could not auto-detect columns. Available columns: {list(df.columns)}. "
            f"Detected text: {detected_text_col}, label: {detected_label_col}"
        )

    return detected_text_col, detected_label_col


def normalize_labels(series: pd.Series) -> pd.Series:
    """Normalize label series to binary values: spam = 1, ham = 0.

    Args:
        series (pd.Series): Raw label column values.

    Returns:
        pd.Series: Normalized integer labels (0 for ham, 1 for spam).

    Raises:
        ValueError: If labels cannot be parsed into binary classes.
    """
    mapping = {
        "spam": 1,
        "ham": 0,
        "1": 1,
        "0": 0,
        1: 1,
        0: 0,
        "true": 1,
        "false": 0,
        "positive": 1,
        "negative": 0,
        "junk": 1,
        "normal": 0
    }

    # Clean strings if series contains objects
    normalized = series.apply(
        lambda x: mapping.get(str(x).strip().lower(), mapping.get(x, None))
    )

    if normalized.isnull().any():
        unknown_values = series[normalized.isnull()].unique()
        raise ValueError(
            f"Found unrecognized label values: {unknown_values}. "
            "Expected binary spam/ham indicators."
        )

    return normalized.astype(int)


def load_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load, inspect, and clean the spam dataset.

    Drops null values and duplicates, and normalizes columns to ['text', 'label'].

    Args:
        file_path (Optional[str]): Explicit path to CSV. If None, auto-searches data/ folder.

    Returns:
        pd.DataFrame: Cleaned dataframe with exactly two columns: 'text' (str) and 'label' (int).

    Raises:
        FileNotFoundError: If no CSV file is found in data/.
    """
    if file_path is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV dataset found in '{data_dir}'. Please place your dataset CSV file there."
            )
        file_path = csv_files[0]

    # Handle common encodings (utf-8, latin-1)
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")

    text_col, label_col = detect_columns(df)

    # Extract only text and label columns
    df_clean = df[[text_col, label_col]].copy()
    df_clean.rename(columns={text_col: "text", label_col: "label"}, inplace=True)

    # Drop nulls and duplicates
    df_clean.dropna(subset=["text", "label"], inplace=True)
    df_clean["text"] = df_clean["text"].astype(str)
    df_clean.drop_duplicates(subset=["text"], inplace=True)

    # Normalize labels
    df_clean["label"] = normalize_labels(df_clean["label"])

    # Reset index
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean
