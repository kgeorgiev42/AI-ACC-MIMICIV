# Functions for data exploration and statistical testing

This document provides an overview of the functions defined in `src.utils.exploration`. Each function is listed with its signature and docstring.

---

## `get_table_one`

```python
def get_table_one(
    ed_pts: pl.DataFrame | pl.LazyFrame,
    outcome: str,
    outcome_label: str,
    output_path: str = "../outputs/reference",
    disp_dict_path: str = "../outputs/reference/feat_name_map.json",
    sensitive_attr_list: list = "None",
    nn_attr: list = "None",
    adjust_method="bonferroni",
    cat_cols: list = None,
    verbose: bool = False,
) -> TableOne:
    """
    Generate a baseline patient summary table (Table 1) with adjusted p-values grouped by outcome.

    Args:
        ed_pts (pl.DataFrame | pl.LazyFrame): Patient data.
        outcome (str): Outcome variable name.
        outcome_label (str): Display name for the outcome.
        output_path (str): Directory to save the HTML summary.
        disp_dict_path (str): Path to JSON mapping feature names to display names.
        sensitive_attr_list (list): List of sensitive attribute names.
        nn_attr (list): List of non-normal columns.
        adjust_method (str): Method for p-value adjustment.
        cat_cols (list): List of categorical columns.
        verbose (bool): If True, print summary information.

    Returns:
        TableOne: Generated TableOne summary object.
    """
```

---

## `assign_age_groups`

```python
def assign_age_groups(
    ed_pts: pl.DataFrame | pl.LazyFrame,
    age_col: str = "anchor_age",
    bins: list = None,
    labels: list = None,
    use_lazy: bool = False,
) -> pl.DataFrame:
    """
    Assign age groups to patients based on age column and specified bins/labels.

    Args:
        ed_pts (pl.DataFrame | pl.LazyFrame): Patient data.
        age_col (str): Name of the age column.
        bins (list): List of bin edges for age groups.
        labels (list): List of labels for age groups.
        use_lazy (bool): If True, return a LazyFrame.

    Returns:
        pl.DataFrame: DataFrame with an added 'age_group' column.
    """
```

---

## `get_age_table_by_sensitive_attr`

```python
def get_age_table_by_sensitive_attr(
    ed_pts: pl.DataFrame | pl.LazyFrame,
    attr_name: str,
    outcome: str,
    value_name_col: str,
) -> pl.DataFrame:
    """
    Transform the dataset into long format to group samples with the outcome by age group and sensitive attribute.

    Args:
        ed_pts (pl.DataFrame | pl.LazyFrame): Patient data.
        attr_name (str): Sensitive attribute column name.
        outcome (str): Outcome variable name.
        value_name_col (str): Name for the value column in the melted DataFrame.

    Returns:
        pd.DataFrame: Long-format DataFrame with counts and percentages by group.
    """
```

---

## `plot_outcome_dist_by_sensitive_attr`

```python
def plot_outcome_dist_by_sensitive_attr(
    ed_pts: pl.DataFrame | pl.LazyFrame,
    attr_col: str,
    attr_xlabel: str,
    output_path: str = "../outputs/reference",
    outcome_list: list = None,
    outcome_title: list = None,
    outcome_legend: dict = None,
    maxi: int = 2,
    maxj: int = 2,
    rot: int = 0,
    figsize: tuple = (8, 6),
    palette: list = None,
):
    """
    Plot the distribution of health outcomes by a specified sensitive attribute.

    Args:
        ed_pts (pl.DataFrame | pl.LazyFrame): Patient data.
        attr_col (str): Sensitive attribute column name.
        attr_xlabel (str): Label for the sensitive attribute display label.
        output_path (str): Directory to save the plot.
        outcome_list (list): List of outcome variable names.
        outcome_title (list): List of outcome display names.
        outcome_legend (dict): Mapping of outcome titles to legend labels.
        maxi (int): Number of rows in subplot grid.
        maxj (int): Number of columns in subplot grid.
        rot (int): Rotation angle for x-axis labels.
        figsize (tuple): Figure size.
        palette (list): List of colors for plotting.

    Returns:
        None
    """
```

---

## `plot_age_dist_by_sensitive_attr`

```python
def plot_age_dist_by_sensitive_attr(
    ed_pts: pl.DataFrame | pl.LazyFrame,
    attr_col: str,
    attr_xlabel: str,
    output_path: str = "../outputs/reference",
    outcome_list: list = None,
    outcome_title: list = None,
    maxi: int = 2,
    maxj: int = 2,
    colors: list = None,
    labels: list = None,
    figsize: tuple = (12, 8),
    rot: int = 0,
):
    """
    Plot the distribution of age groups by a specified sensitive attribute for patients with adverse events.

    Args:
        ed_pts (pl.DataFrame | pl.LazyFrame): Patient data.
        attr_col (str): Sensitive attribute column name.
        attr_xlabel (str): Label for the sensitive attribute display name.
        output_path (str): Directory to save the plot.
        outcome_list (list): List of outcome variable names.
        outcome_title (list): List of outcome display names.
        maxi (int): Number of rows in subplot grid.
        maxj (int): Number of columns in subplot grid.
        colors (list): List of colors for age groups.
        labels (list): List of age group labels.
        figsize (tuple): Figure size.
        rot (int): Rotation angle for x-axis labels.

    Returns:
        None
    """
```

---

## `plot_token_length_by_attribute`

```python
def plot_token_length_by_attribute(
    ed_pts: pl.DataFrame | pl.LazyFrame,
    output_path: str = "../outputs/reference",
    sensitive_attr_list: list = None,
    attr_title: list = None,
    out_fname: str = "bhc_dist_by_attr.png",
    maxi: int = 2,
    maxj: int = 2,
    figsize: tuple = (8, 6),
    rot: int = 0,
    ylim: tuple = (0, 12),
    gr_pairs: dict = None,
    suptitle: str = "BHC token length by sensitive variable in patients with ED attendance.",
    outcome_mode: bool = False,
    unique_value_order: list = None,
    adjust_method: str = "bonferroni",
    test_type: str = "t-test_welch",
):
    """
    Display violin plots of aggregated BHC length by sensitive attribute, with statistical annotation.

    Args:
        ed_pts (pl.DataFrame | pl.LazyFrame): Patient data.
        output_path (str): Directory to save the plot.
        sensitive_attr_list (list): List of sensitive attribute names.
        attr_title (list): List of attribute display names.
        out_fname (str): Output filename for the plot.
        maxi (int): Number of rows in subplot grid.
        maxj (int): Number of columns in subplot grid.
        figsize (tuple): Figure size.
        rot (int): Rotation angle for x-axis labels.
        ylim (tuple): Y-axis limits.
        gr_pairs (dict): Dictionary of group pairs for statistical annotation.
        suptitle (str): Figure title.
        outcome_mode (bool): If True, relabel categories for outcome mode.
        unique_value_order (list): Order of unique values for plotting.
        adjust_method (str): Method for p-value adjustment.
        test_type (str): Statistical test type.

    Returns:
        None
    """
```

---
