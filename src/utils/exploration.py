import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.patches import Patch
from statannotations.Annotator import Annotator
from tableone import TableOne


def get_table_one(
    ed_pts: pl.DataFrame | pl.LazyFrame,
    ts_measures: pl.DataFrame | pl.LazyFrame,
    outcome: str,
    outcome_label: str,
    output_path: str = "../outputs/reference",
    disp_dict_path: str = "../outputs/reference/feat_name_map.json",
    sensitive_attr_list: list = "None",
    nn_attr: list = "None",
    n_attr: list = "None",
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
        n_attr (list): List of normal columns.
        adjust_method (str): Method for p-value adjustment.
        cat_cols (list): List of categorical columns.
        verbose (bool): If True, print summary information.

    Returns:
        TableOne: Generated TableOne summary object.
    """
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect().to_pandas()
        ts_measures = ts_measures.collect(streaming=True).to_pandas()
    else:
        ed_pts = ed_pts.to_pandas()
        ts_measures = ts_measures.to_pandas()
    ### Load dictionary containing display names for features
    with open(disp_dict_path) as f:
        disp_dict = json.load(f)
    ### Infer categorical data if not specified
    if cat_cols is None:
        cat_cols = [el for el in disp_dict.values() if el not in nn_attr and el not in n_attr]

    ## Create handles for prescriptions
    ed_pts['presc_ACEi'] = np.where(ed_pts['n_presc_acei'] > 0, 'Yes', 'No')
    ed_pts['presc_ARB'] = np.where(ed_pts['n_presc_arb'] > 0, 'Yes', 'No')
    ed_pts['presc_BB'] = np.where(ed_pts['n_presc_bb'] > 0, 'Yes', 'No')
    ed_pts['presc_DAPT'] = np.where(ed_pts['n_presc_dapt'] > 0, 'Yes', 'No')
    ed_pts['presc_Aspirin'] = np.where(ed_pts['n_presc_aspirin'] > 0, 'Yes', 'No')

    ed_pts = ed_pts.rename(columns={'presc_ACEi': 'ACE Inhibitors',
        'presc_ARB': 'Angiotensin II Receptor Blockers',
        'presc_BB': 'Beta Blockers',
        'presc_DAPT': 'Dual Antiplatelet Therapy',
        'presc_Aspirin': 'Aspirin'})
    presc_cols = ['ACE Inhibitors',
        'Angiotensin II Receptor Blockers', 'Beta Blockers',
        'Dual Antiplatelet Therapy', 'Aspirin']

    cat_cols = cat_cols + presc_cols
    full_cols = list(disp_dict.values()) + presc_cols
    full_cols = [col for col in full_cols if 'Diagnosed' not in col and 'Suspected' not in col]
    full_cols = full_cols + [outcome_label]

    ## Handle time-series measurements: get median values per patient
    ts_median_df = get_median_values_per_patient(ts_measures)
    # Merge the pivoted DataFrame with ed_pts
    ed_pts = ed_pts.merge(ts_median_df, on='subject_id', how='left')
    ed_disp = ed_pts.rename(columns=disp_dict)
    ed_disp = ed_disp[full_cols]
    sensitive_attr_list = sensitive_attr_list + ['Mode of ED Arrival']
    ### Code categorical columns
    for col in ed_disp.columns:
        if (col not in sensitive_attr_list and col not in presc_cols and col in cat_cols) or col == outcome_label:
            ed_disp[col] = np.where(ed_disp[col] == 1, "Yes", "No")
    ### Generate Table 1 summary with p-values for target outcome
    cat_cols = [col for col in cat_cols if col in ed_disp.columns and col != outcome_label]

    if verbose:
        prevalence = round(len(ed_disp[ed_disp[outcome_label] == "Yes"]) / ed_disp.shape[0] * 100, 2)
        print(
            f"Generating table summary by {outcome_label} with prevalence {prevalence}%"
        )

        # Set decimals for specific variables
        decimals_dict = {"Temperature (deg-C)": 2,
            "Systolic Blood Pressure (mmHg)": 2,
            "Heart Rate (bpm)": 2,
            "Oxygen Saturation (%)": 2,
            "Respiratory Rate (breaths/min)": 2,
            "Hemoglobin (g/dL)": 2,
            "eGFR (mL/min/1.73msq)": 2,
            "Troponin T (ng/L)": 2,
            "White Blood Cell Count (10^9/L)": 2,
            "Creatinine (micmol/L)": 2,
            "Urea Nitrogen (mmol/L)": 2,
            "Hematocrit (%)": 2,
            "Platelet Count (10^9/L)": 2,
            "Potassium (mmol/L)": 2,
            "Sodium (mmol/L)": 2,
            "Chloride (mmol/L)": 2,
            "Bicarbonate (mmol/L)": 2,
            "Anion Gap (mmol/L)": 2,
            "Creatine Kinase-MB (ng/mL)": 2}

    # Set decimals=2 for specified variables, 0 for the rest
    decimals = {col: decimals_dict.get(col, 0) for col in ed_disp.columns}

    tb_one_hd = TableOne(
        ed_disp,
        categorical=cat_cols,
        nonnormal=nn_attr,
        groupby=outcome_label,
        overall=True,
        pval=True,
        htest_name=True,
        tukey_test=True,
        decimals=decimals,
        pval_adjust=adjust_method,
    )
    tb_one_hd.to_html(os.path.join(output_path, f"table_one_{outcome}.html"))
    print(
        f"Saved table summary grouped by {outcome_label} to table_one_{outcome}.html."
    )
    return tb_one_hd


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
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect()
    ed_pts = ed_pts.with_columns(
        pl.when(pl.col(age_col) < bins[1])
        .then(pl.lit(labels[0]))
        .when((pl.col(age_col) >= bins[1]) & (pl.col(age_col) < bins[2]))
        .then(pl.lit(labels[1]))
        .when((pl.col(age_col) >= bins[2]) & (pl.col(age_col) < bins[3]))
        .then(pl.lit(labels[2]))
        .when((pl.col(age_col) >= bins[3]) & (pl.col(age_col) < bins[4]))
        .then(pl.lit(labels[3]))
        .otherwise(pl.lit(labels[4]))
        .alias("age_group")
    )
    return ed_pts.lazy() if use_lazy else ed_pts


def get_median_values_per_patient(ed_ts_measures: pd.DataFrame | pl.DataFrame | pl.LazyFrame) -> pd.DataFrame:
    """
    Extract the median value per patient for each measurement label from a long-format time-series dataframe.

    Args:
        ed_ts_measures (pd.DataFrame | pl.DataFrame | pl.LazyFrame): Time-series measurements in long format with 'subject_id', 'label', and 'value'.

    Returns:
        pd.DataFrame: DataFrame with columns ['subject_id', 'label', 'median_value'].
    """
    # Convert to pandas if it's a Polars DataFrame
    if isinstance(ed_ts_measures, pl.LazyFrame):
        ed_ts_measures = ed_ts_measures.collect().to_pandas()
    elif isinstance(ed_ts_measures, pl.DataFrame):
        ed_ts_measures = ed_ts_measures.to_pandas()

    # Use pandas groupby and pivot
    median_df = ed_ts_measures.groupby(["subject_id", "label"]).agg(
        median_value=("value", "median")
    ).reset_index()

    # Pivot table by label
    median_df = median_df.pivot(index="subject_id", columns="label", values="median_value").reset_index()

    return median_df
