# Generic functions for manipulating Python files and objects

This document provides an overview of the functions defined in `src.utils.functions`. Each function is listed with its signature and docstring.

---

## `load_pickle`

```python
def load_pickle(filepath: str) -> Any:
    """Load a pickled object.

    Args:
        filepath (str): Path to pickle (.pkl) file.

    Returns:
        Any: Loaded object.
    """
```

---

## `save_pickle`

```python
def save_pickle(target: dict, filepath: str, fname: str = "mm_feat.pkl") -> Any:
    """
    Save a Python object as a pickle file.

    Args:
        target (dict): Object to pickle.
        filepath (str): Directory to save the pickle file.
        fname (str): Filename for the pickle file (default: "mm_feat.pkl").

    Returns:
        None
    """
```

---

## `impute_from_df`

```python
def impute_from_df(
    impute_to: pl.DataFrame | pl.LazyFrame,
    impute_from: pl.DataFrame,
    use_col: str = None,
    key_col: str = None,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Impute values from one dataframe to another using a key column.

    Args:
        impute_to (pl.DataFrame | pl.LazyFrame): Table to impute values into.
        impute_from (pl.DataFrame): Table to impute values from.
        use_col (str, optional): Column containing values to impute.
        key_col (str, optional): Column to use to identify matching rows.

    Returns:
        pl.DataFrame | pl.LazyFrame: DataFrame with imputed values.
    """
```

---

## `get_final_episodes`

```python
def get_final_episodes(stays: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Extracts the final ED episode with hospitalisation for creating a unique patient cohort.

    Args:
        stays (pl.DataFrame): Stays data.

    Returns:
        pl.DataFrame: Patient-level data.
    """
```

---

## `get_n_unique_values`

```python
def get_n_unique_values(
    table: pl.DataFrame | pl.LazyFrame, use_col: str = "subject_id"
) -> int:
    """Compute number of unique values in particular column in table.

    Args:
        table (pl.DataFrame | pl.LazyFrame): Table.
        use_col (str, optional): Column to use. Defaults to "subject_id".

    Returns:
        int: Number of unique values.
    """
```

---

## `scale_numeric_features`

```python
def scale_numeric_features(
    table: pl.DataFrame, numeric_cols: list = None, over: str = None
) -> pl.DataFrame:
    """Applies min/max scaling to numeric columns and rounds to 1 d.p.

    Args:
        table (pl.DataFrame): Table.
        numeric_cols (list, optional): List of columns to apply to. Defaults to None.
        over (str, optional): Column to group by before computing min/max. Defaults to None.

    Returns:
        pl.DataFrame: Updated table.
    """
```

---

## `read_icd_mapping`

```python
def read_icd_mapping(map_path: str) -> pl.DataFrame:
    """
    Reads ICD-9 to ICD-10 mapping file for chronic conditions.
    """
```

---

## `contains_both_ltc_types`

```python
def contains_both_ltc_types(ltc_set: set) -> bool:
    """
    Helper util function for physical-mental multimorbidity detection.

    Args:
        ltc_set (set): Set containing LTC codes.

    Returns:
        bool: True if both physical and mental LTC types are present, False otherwise.
    """
```

---

## `preview_data`

```python
def preview_data(filepath: str) -> None:
    """Prints a single example from data dictionary.

    Args:
        filepath (str): Path to .pkl file containing data dictionary.
    """
```

---

## `get_demographics_summary`

```python
def get_demographics_summary(ed_pts: pl.DataFrame | pl.LazyFrame) -> None:
    """
    Summarises sensitive attributes and outcome prevalence.
    Args:
        demographics (pl.DataFrame): Demographics data.

    Returns:
        pl.DataFrame: Summary table.
    """
```

---

## `get_train_split_summary`

```python
def get_train_split_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    outcome: str = "in_hosp_death",
    output_path: str = "../outputs/exp_data",
    cont_cols: list = None,
    nn_cols: list = None,
    disp_dict: dict = None,
    cat_cols: list = None,
    verbose: bool = True,
) -> None:
    """
    Print and save a statistical summary for the train, validation, and test splits.

    Args:
        train (pd.DataFrame): Training set DataFrame.
        val (pd.DataFrame): Validation set DataFrame.
        test (pd.DataFrame): Test set DataFrame.
        outcome (str): Name of the outcome variable (default: "in_hosp_death").
        output_path (str): Directory to save the summary HTML file.
        cont_cols (list): List of continuous columns.
        nn_cols (list): List of non-normal columns.
        disp_dict (dict): Dictionary mapping original to display column names.
        cat_cols (list): List of categorical columns.
        verbose (bool): If True, print progress messages.

    Returns:
        None
    """
```

---

## `rename_fields`

```python
def rename_fields(col):
    """
    Helper function to rename drug and specialty feature names.

    Args:
        col (Any): Column name or tuple of column names.

    Returns:
        str: Joined string if input is a tuple, otherwise the original column name.
    """
```

---

## `read_from_txt`

```python
def read_from_txt(filepath: str, as_type="str") -> list:
    """Read from line-seperated txt file.

    Args:
        filepath (str): Path to text file.

    Returns:
        list: List containing data.
    """
```
