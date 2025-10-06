# Functions describing the explainability analysis pipeline with SHAP and MM-SHAP attribution scoring, supporting global and local-level explanations

This document provides an overview of the functions defined in `src.utils.shap_utils`. Each function is listed with its signature and docstring.

---

## `get_feature_names`

```python
def get_feature_names(test_set, modalities):
    """
    Get feature names for each modality in the test set and save the column mappings to a dictionary.

    Args:
        test_set (MIMIC4Dataset): The test set object containing feature information.
        modalities (list): List of modality types ('static', 'timeseries', 'notes').

    Returns:
        dict: Mapping from modality type to list of feature names.
    """
```

---

## `ModelWrapper`

```python
class ModelWrapper(torch.nn.Module):
    """
    A DeepSHAP wrapper around the PyTorch model to ensure scalar outputs for SHAP.

    Args:
        model: The base model to wrap.
        modality (str): The modality type ('static', 'timeseries', 'notes').
        total_dim (int): Input dimension for the linear layer.
        target_size (int): Output dimension for the linear layer.
        ts_ind (int, optional): Target for timeseries data (0 - vital signs, 1 - lab measurements).
    """
```

---

## `get_shap_values`

```python
def get_shap_values(model, batch, device, num_ts, modalities):
    """
    Batch-wise SHAP computation of attribution scores for each modality using the model's prepare_batch method.

    Args:
        model (MMModel): The model to explain.
        batch (DataLoader object): Batch of input data.
        device (torch.device): Torch device ('cpu' or 'gpu').
        num_ts (int): Number of timeseries modalities.
        modalities (list): List of modalities to explain.

    Returns:
        dict: SHAP values for each modality.
    """
```

---

## `estimate_mm_summary`

```python
def estimate_mm_summary(shap_scores, shap_expected_scores):
    """
    Estimate aggregate multimodal SHAP values for calculating relative degree of modality dependence.
    Aggregate SHAP values are inspired by MM-SHAP (https://github.com/Heidelberg-NLP/MM-SHAP).

    Args:
        shap_scores (list): List of SHAP value arrays for each modality.
        shap_expected_scores (list): List of expected (reference) SHAP values for each modality. Estimated from the batch-wise SHAP mean.

    Returns:
        tuple: (mm_scores, shap_expected_ovr, shap_max_ovr, shap_min_ovr)
        mm_scores (list): Multimodal SHAP scores for each modality.
        shap_expected_ovr (float): Overall expected SHAP value across modalities.
        shap_max_ovr (float): Maximum SHAP value across all modalities.
        shap_min_ovr (float): Minimum SHAP value across all modalities.
    """
```

---

## `get_shap_summary_plot`

```python
def get_shap_summary_plot(
    shap_obj,
    outcome="In-hospital Death",
    fusion_type=None,
    modality=None,
    max_features=20,
    figsize=(7, 8),
    save_path=None,
    heatmap=False,
):
    """
    Generate and save a global-level SHAP summary plot for a given modality.

    Args:
        shap_obj: SHAP explanation object.
        outcome (str): Outcome name for plot title.
        fusion_type (str): Fusion type for plot title.
        modality (str): Modality type ('static', 'timeseries', 'notes').
        max_features (int): Maximum number of features to display.
        figsize (tuple): Figure size.
        save_path (str): Path to save the plot.
        heatmap (bool): If True, plot as heatmap.

    Returns:
        None
    """
```

---

## `aggregate_ts`

```python
def aggregate_ts(data):
    """
    Aggregate timeseries SHAP values using mean pooling, ignoring missing values.

    Args:
        data (list or np.ndarray): List of timeseries SHAP arrays.

    Returns:
        np.ndarray: Aggregated SHAP values.
    """
```

---

## `get_shap_local_decision_plot`

```python
def get_shap_local_decision_plot(
    shap_obj,
    risk_quantile=None,
    figsize=(6, 7),
    save_static_path=None,
    save_ts0_path=None,
    save_ts1_path=None,
    save_nt_path=None,
    shap_range=None,
    mm_scores=None,
):
    """
    Generate and save SHAP local-level decision plots for each modality and text highlight plot for the note segments.

    Args:
        shap_obj: List of SHAP explanation objects.
        risk_quantile: Risk quantile for plot title.
        figsize (tuple): Figure size.
        save_static_path (str): Path to save static modality plot.
        save_ts0_path (str): Path to save timeseries vitals plot.
        save_ts1_path (str): Path to save timeseries labs plot.
        save_nt_path (str): Path to save notes plot.
        shap_range (tuple): SHAP value range for plots.
        mm_scores (list): Multimodal SHAP scores.

    Returns:
        None
    """
```

---

## `_draw_token`

```python
def _draw_token(ax, text, x, y, color, max_x, line_height):
    """
    Draw a single sentence token from a discharge sumamry with background color on the plot.

    Args:
        ax: Matplotlib axis.
        text (str): Token text.
        x (float): X position.
        y (float): Y position.
        color: Background color.
        max_x (float): Maximum X position.
        line_height (float): Height of a line.

    Returns:
        tuple: Updated (x, y) positions.
    """
```

---

## `_draw_next_note`

```python
def _draw_next_note(ax, x, y, line_height, note_text):
    """
    Draw a separator for the next note in the plot.

    Args:
        ax: Matplotlib axis.
        x (float): X position.
        y (float): Y position.
        line_height (float): Height of a line.
        note_text (str): Separator text.

    Returns:
        tuple: Updated (x, y) positions.
    """
```

---

## `_process_token_line`

```python
def _process_token_line(ax, token_line, val, cmap, norm, x, y, max_x, line_height):
    """
    Process and draw a line of tokens, handling note separators.

    Args:
        ax: Matplotlib axis.
        token_line (str): Line of tokens.
        val (float): SHAP value.
        cmap: Colormap.
        norm: Normalization for colormap.
        x (float): X position.
        y (float): Y position.
        max_x (float): Maximum X position.
        line_height (float): Height of a line.

    Returns:
        tuple: Updated (x, y, token_line).
    """
```

---

## `_render_highlighted_text`

```python
def _render_highlighted_text(
    ax, shap_values, text_tokens, norm, cmap, line_height=0.08, max_x=0.98
):
    """
    Render highlighted text with color background for SHAP values.

    Args:
        ax: Matplotlib axis.
        shap_values (np.ndarray): SHAP values for each token.
        text_tokens (list): List of text tokens.
        norm: Normalization for colormap.
        cmap: Colormap.
        line_height (float): Height of a line.
        max_x (float): Maximum X position.

    Returns:
        float: Final Y position after rendering.
    """
```

---

## `plot_highlighted_text_with_colorbar`

```python
def plot_highlighted_text_with_colorbar(
    shap_values,
    text_tokens,
    expected_value,
    mm_score,
    figsize=(10, 4),
    cmap="coolwarm",
    save_path=None,
    shap_range=None,
):
    """
    Display a highlighted text plot and a colorbar based on SHAP values.

    Args:
        shap_values (np.ndarray): SHAP values for each token (1D array).
        text_tokens (list of str): List of text tokens (words or sentences).
        expected_value (float): Reference value for colorbar label.
        mm_score (float): MM-SHAP dependence score for notes modality.
        figsize (tuple): Figure size.
        cmap (str): Matplotlib colormap.
        save_path (str): If provided, saves the plot to this path.
        shap_range (tuple): Min/Max SHAP values for colorbar.

    Returns:
        None
    """
```

---
