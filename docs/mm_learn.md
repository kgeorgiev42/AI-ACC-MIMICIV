# PyTorch pipeline for multimodal learning

This document provides an overview of the classes and functions defined in `src.models`. Each class or function is listed with its signature and docstring.

---

## `LSTM`

```python
class LSTM(nn.Module):
    """
    General-purpose LSTM module for sequence embeddings.

    Args:
        input_dim (int): Input feature dimension.
        embed_dim (int): Output embedding dimension.
        num_layers (int): Number of LSTM layers.
        hidden_dim (int): Hidden state dimension.
        dropout (float): Dropout rate.

    Returns:
        torch.Tensor: Embedded output.
    """
```

---

## `Gate`

```python
class Gate(nn.Module):
    """
    Gated fusion module for combining multiple input embeddings.
    Adapted from https://github.com/emnlp-mimic/mimic/blob/main/base.py#L136 inspired by https://arxiv.org/pdf/1908.05787.
    Args:
        inp1_size (int): Size of first input.
        inp2_size (int): Size of second input.
        inp3_size (int): Size of third input (optional).
        dropout (float): Dropout rate.

    Returns:
        torch.Tensor: Fused output.
    """
```

---

## `GradientReversalFunction`

```python
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal layer for adversarial training.

    Used to reverse gradients during backpropagation for adversarial debiasing.
    """
```

---

## `grad_reverse`

```python
def grad_reverse(x, lambda_=1.0):
    """
    Apply gradient reversal to the input tensor to enable maximisation of the adversarial objective function.

    Args:
        x (torch.Tensor): Input tensor.
        lambda_ (float): Scaling factor for gradient reversal.

    Returns:
        torch.Tensor: Output tensor with reversed gradients.
    """
```

---

## `MMModel`

```python
class MMModel(L.LightningModule):
    """
    Multimodal model object for fusion of static, timeseries, and notes data.

    Args:
        st_input_dim (int): Static input dimension.
        st_embed_dim (int): Static embedding dimension.
        ts_input_dim (tuple): Tuple of timeseries input dimensions.
        ts_embed_dim (int): Timeseries embedding dimension.
        nt_input_dim (int): Notes input dimension.
        nt_embed_dim (int): Notes embedding dimension.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate.
        num_ts (int): Number of timeseries modalities.
        target_size (int): Output size.
        lr (float): Learning rate.
        fusion_method (str): Fusion method ("concat" or "mag").
        st_first (bool): If True, static features are fused first.
        modalities (list): List of modalities to use.
        with_packed_sequences (bool): If True, use packed sequences for timeseries.
        dataset (MIMIC4Dataset): Pass training dataset if using class weighting, else None.
        sensitive_attr_ids (list): Indices of sensitive attribute features from static data for adversarial debiasing.
        adv_lambda (float): Strength of adversarial penalty. No penalty is 0. Slight penalty is 0.1-0.2. Strong penalty is >=1.

    Returns:
        torch.Tensor: Model output.
    """
```

---

## `LitLSTM`

```python
class LitLSTM(L.LightningModule):
    """
    LSTM model for time-series data only.

    Args:
        ts_input_dim (int): Timeseries input dimension.
        lstm_embed_dim (int): LSTM embedding dimension.
        target_size (int): Output size.
        lr (float): Learning rate.
        with_packed_sequences (bool): If True, use packed sequences.

    Returns:
        torch.Tensor: Model output.
    """
```

---

## `SaveLossesCallback`

```python
class SaveLossesCallback(Callback):
    """
    Learner callback to save train/validation losses to a CSV file.
    """
```

---
