# Functions describing the multimodal evaluation pipeline for assessing model performance and performing risk stratification

This document provides an overview of the functions defined in `src.utils.eval_utils`. Each function is listed with its signature and docstring.

---

## `plot_learning_curve`

```python
def plot_learning_curve(losses_path: str = None, output_path="learning_curve.png"):
    """
    Plot the learning curve (training and validation loss) from a CSV file.

    Args:
        losses_path (str): Path to CSV file containing the training and validation loss.
        output_path (str): Path to save the learning curve plot.

    Returns:
        None
    """
```

---

## `plot_roc`

```python
def plot_roc(
    y_test: np.array,
    prob: np.array,
    output_path: str = "roc.png",
    result_dict: dict = None,
    outcome: str = "In-hospital Death",
):
    """
    Plot the ROC curve with AUC and 95% CI for a binary classifier.

    Args:
        y_test (np.array): Ground truth binary labels.
        prob (np.array): Predicted probabilities for the positive class.
        output_path (str): Path to save the ROC curve plot.
        result_dict (dict): Target performance dictionary for storing performance metrics.
        outcome (str): Name of the outcome being evaluated.

    Returns:
        None
    """
```

---

## `plot_pr`

```python
def plot_pr(
    y_test: np.array,
    prob: np.array,
    output_path: str = "pr_curve.png",
    result_dict: dict = None,
    outcome: str = "In-hospital Death",
):
    """
    Plot the Precision-Recall curve with AUC and 95% CI for a binary classifier.

    Args:
        y_test (np.array): Ground truth binary labels.
        prob (np.array): Predicted probabilities for the positive class.
        output_path (str): Path to save the PR curve plot.
        result_dict (dict): Target performance dictionary for storing performance metrics.
        outcome (str): Name of the outcome being evaluated.

    Returns:
        None
    """
```

---

## `plot_calibration_curve`

```python
def plot_calibration_curve(
    y_test: np.array,
    prob: np.array,
    output_path: str = "calib_curve.png",
    outcome: str = "In-hospital Death",
    n_bins: int = 10,
):
    """
    Plot the calibration curve for a binary classifier.

    Args:
        y_test (np.array): Ground truth binary labels (0 or 1).
        prob (np.array): Predicted probabilities for the positive class.
        output_path (str): Path to save the calibration curve plot.
        outcome (str): Name of the outcome being evaluated.
        n_bins (int): Number of bins to use for calibration.

    Returns:
        None
    """
```

---

## `expect_f1`

```python
def expect_f1(y_prob: np.array, thres: int) -> float:
    """
    Calculate expected F1 score for a given threshold.

    Args:
        y_prob (np.array): Predicted probabilities.
        thres (float): Threshold for binary classification.

    Returns:
        float: Expected F1 score.
    """
```

---

## `optimal_threshold`

```python
def optimal_threshold(y_prob: np.array) -> float:
    """
    Calculate the optimal threshold for binary classification based on expected F1 score.

    Args:
        y_prob (np.array): Predicted probabilities.

    Returns:
        float: Optimal threshold.
    """
```

---

## `get_roc_performance`

```python
def get_roc_performance(y_test: np.array, prob: np.array, verbose: bool = False):
    """
    Compute ROC performance summary based on Youden's J statistic for a binary classifier.

    Args:
        y_test (np.array): Ground truth binary labels.
        prob (np.array): Predicted probabilities for the positive class.
        verbose (bool): If True, print detailed performance metrics.

    Returns:
        tuple: (bin_labels, res_dict_roc)
            bin_labels (np.array): Binary predictions using Youden's J threshold.
            res_dict_roc (dict): ROC statistics and confidence intervals.
    """
```

---

## `get_pr_performance`

```python
def get_pr_performance(
    y_test: np.array,
    prob: np.array,
    bin_labels: np.array,
    opt_f1: bool = True,
    verbose: bool = False,
):
    """
    Compute Precision-Recall performance metrics and confidence intervals.

    Args:
        y_test (np.array): Ground truth binary labels.
        prob (np.array): Predicted probabilities for the positive class.
        bin_labels (np.array): Binary predictions.
        opt_f1 (bool): If True, use optimal F1 threshold.
        verbose (bool): If True, print detailed performance metrics.

    Returns:
        dict: PR statistics and confidence intervals.
    """
```

---

## `get_all_roc_pr_summary`

```python
def get_all_roc_pr_summary(
    res_dicts: list,
    models: list,
    colors: list,
    output_roc_path: str = "roc_summary.png",
    output_pr_path: str = "pr_summary.png",
):
    """
    Plot summary ROC and PR curves for multiple models.

    Args:
        res_dicts (list): List of dictionaries with model results.
        models (list): List of model names.
        colors (list): List of colors for plotting.
        output_roc_path (str): Path to save ROC summary plot.
        output_pr_path (str): Path to save PR summary plot.

    Returns:
        None
    """
```

---

## `rank_prediction_quantiles`

```python
def rank_prediction_quantiles(
    y_test: np.array,
    prob: np.array,
    attrs: list,
    attr_disp: list,
    test_ids: list,
    n_bins: int = 10,
    outcome: str = "In-hospital Death",
    output_path: str = "risk_strat.png",
    by_attribute: bool = False,
    attr_features: pd.DataFrame = None,
    verbose: bool = False,
):
    """
    Rank predictions into quantiles and plot risk stratification, optionally stratified by attribute.

    Args:
        y_test (np.array): Ground truth binary labels.
        prob (np.array): Predicted probabilities for the positive class.
        attrs (list): List of attribute names for stratification.
        attr_disp (list): List of display names for attributes.
        test_ids (list): List of subject IDs for test set.
        n_bins (int): Number of quantiles.
        outcome (str): Name of the outcome being evaluated.
        output_path (str): Path to save the risk stratification plot.
        by_attribute (bool): If True, stratify by attribute.
        attr_features (pd.DataFrame): DataFrame with attribute features.
        verbose (bool): If True, print detailed information.

    Returns:
        dict: Appended patient risk quantiles.
    """
```

---
