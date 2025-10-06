# Functions describing the fairness evaluation pipeline with added bootstrapping

This document provides an overview of the functions defined in `src.utils.fairness_utils`. Each function is listed with its signature and docstring.

---

## `plot_bar_metric_frame`

```python
def plot_bar_metric_frame(
    metrics: dict,
    y_test: np.ndarray,
    y_hat: np.ndarray,
    attr_df: pl.DataFrame,
    attribute: str,
    save_path: str,
    figsize=None,
    nrows=2,
    ncols=2,
    seed=0,
):
    """
    Plot an error bar chart for the given metric frame using Fairlearn.

    Args:
        metrics (dict): Dictionary of metric functions to compute.
        y_test (np.ndarray): Ground truth labels.
        y_hat (np.ndarray): Predicted labels.
        attr_df (pl.DataFrame): Sensitive attribute DataFrame.
        attribute (str): Name of the sensitive attribute.
        save_path (str): Path to save the plot.
        figsize (tuple, optional): Figure size.
        nrows (int): Number of subplot rows.
        ncols (int): Number of subplot columns.
        seed (int): Random seed.

    Returns:
        None
    """
```

---

## `bias_corrected_ci`

```python
def bias_corrected_ci(bootstrap_samples, observed_value):
    """
    Calculate bias-corrected and accelerated (BCa) confidence intervals for bootstrap samples.

    Args:
        bootstrap_samples (array-like): Bootstrap sample values.
        observed_value (float): Observed value for bias correction.

    Returns:
        tuple: (lower_bound, upper_bound) confidence interval.
    """
```

---

## `get_bootstrapped_fairness_measures`

```python
def get_bootstrapped_fairness_measures(
    y_test: np.ndarray,
    y_hat: np.ndarray,
    attr_pf: pl.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
    skip_ci: bool = False,
    verbose: bool = False,
) -> tuple:
    """
    Compute bootstrapped fairness measures (Demographic Parity, Equalized Odds, Equal Opportunity)
    and their confidence intervals.

    Args:
        y_test (np.ndarray): Ground truth labels.
        y_hat (np.ndarray): Predicted labels.
        attr_pf (pl.DataFrame): Sensitive attribute DataFrame.
        n_boot (int): Number of bootstrap samples.
        seed (int): Random seed.
        skip_ci (bool): If True, skip CI calculation.
        verbose (bool): If True, print additional information.

    Returns:
        tuple: (dpr_full, eor_full, eop_full) where each contains (mean, lower_CI, upper_CI).
    """
```

---

## `plot_fairness_by_age`

```python
def plot_fairness_by_age(
    aq_dict: dict,
    age_labels: list,
    out_path: str,
    attributes: list,
    attribute_labels: list,
    figsize: tuple = (11, 8),
    measure: str = "DPR",
    measure_label: str = "Demographic Parity",
):
    """
    Plot fairness measures by age group for multiple sensitive attributes.

    Args:
        aq_dict (dict): Dictionary containing fairness metrics by age group.
        age_labels (list): List of age group labels.
        out_path (str): Path to save the plot.
        attributes (list): List of sensitive attribute names.
        attribute_labels (list): List of attribute display names.
        figsize (tuple): Figure size.
        measure (str): Key for the fairness measure to plot.
        measure_label (str): Display label for the fairness measure.

    Returns:
        None
    """
```

---

## `get_fairness_summary`

```python
def get_fairness_summary(
    res_all: dict,
    models: list,
    colors: list,
    attribute_labels: list,
    figsize: tuple = (13, 8),
    nrows: int = 2,
    ncols: int = 2,
    outcome: str = "Extended Stay",
    output_path: str = "fair_full_across_models.png",
):
    """
    Plot grouped barplots for fairness metrics (DPR, EQO, EOP) with 95% CI for each model.

    Args:
        res_all (dict): Dictionary containing fairness metrics for each model.
        models (list): List of model names.
        colors (list): List of colors for each model.
        attribute_labels (list): List of attribute display names.
        figsize (tuple): Figure size.
        nrows (int): Number of subplot rows.
        ncols (int): Number of subplot columns.
        outcome (str): Name of the outcome.
        output_path (str): Path to save the plot.

    Returns:
        None
    """
```

---
