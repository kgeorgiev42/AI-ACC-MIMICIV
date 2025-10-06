# Functions for preprocessing and cleaning extracted data, as well as generating multimodal features

This document provides an overview of the functions defined in `src.utils.preprocessing`. Each function is listed with its signature and docstring.

---

## `preproc_icd_module`

```python
def preproc_icd_module(
    diagnoses: pl.DataFrame | pl.LazyFrame,
    icd_map_path: str = "../config/icd9to10.txt",
    map_code_colname: str = "diagnosis_code",
    only_icd10: bool = True,
    ltc_dict_path: str = "../outputs/icd10_codes.json",
    verbose=True,
    use_lazy: bool = False,
) -> pl.DataFrame:
    """
    Process a diagnoses dataset with ICD codes, mapping ICD-9 to ICD-10 and generating features for long-term conditions.
    Implementation is taken from the MIMIC-IV preprocessing pipeline provided by Gupta et al. (https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/tree/main).

    Args:
        diagnoses (pl.DataFrame | pl.LazyFrame): Diagnoses data.
        icd_map_path (str): Path to ICD-9 to ICD-10 mapping file.
        map_code_colname (str): Column name for ICD code in mapping.
        only_icd10 (bool): If True, only keep ICD-10 codes.
        ltc_dict_path (str): Path to JSON with LTC code groups.
        verbose (bool): If True, print summary statistics.
        use_lazy (bool): If True, return a LazyFrame.

    Returns:
        pl.DataFrame or pl.LazyFrame: Processed diagnoses data.
    """
```

---

## `get_ltc_features`

```python
def get_ltc_features(
    admits_last: pl.DataFrame | pl.LazyFrame,
    diagnoses: pl.DataFrame | pl.LazyFrame,
    ltc_dict_path: str = "../outputs/icd10_codes.json",
    mm_cutoff: int = 1,
    cmm_cutoff: int = 3,
    verbose=True,
    use_lazy: bool = False,
) -> pl.DataFrame:
    """
    Generate features for long-term conditions and multimorbidity from ICD-10 diagnoses and custom LTC dictionary.

    Args:
        admits_last (pl.DataFrame | pl.LazyFrame): Admissions data.
        diagnoses (pl.DataFrame | pl.LazyFrame): ICD-10 Diagnoses data.
        ltc_dict_path (str): Path to JSON with LTC code groups.
        mm_cutoff (int): Threshold for multimorbidity.
        cmm_cutoff (int): Threshold for complex multimorbidity.
        verbose (bool): If True, print summary statistics.
        use_lazy (bool): If True, return a LazyFrame.

    Returns:
        pl.DataFrame or pl.LazyFrame: Admissions data with long-term condition count features.
    """
```

---

## `transform_sensitive_attributes`

```python
def transform_sensitive_attributes(ed_pts: pl.DataFrame) -> pl.DataFrame:
    """
    Map sensitive attributes (race, marital status) to predefined categories and types.

    Args:
        ed_pts (pl.DataFrame): Patient data.

    Returns:
        pl.DataFrame: Updated patient data.
    """
```

---

## `prepare_medication_features`

```python
def prepare_medication_features(
    medications: pl.DataFrame | pl.LazyFrame,
    admits_last: pl.DataFrame | pl.LazyFrame,
    top_n: int = 50,
    use_lazy: bool = False,
) -> pl.DataFrame:
    """
    Generate count and temporal (days since prescription) features for drug-level medication history.

    Args:
        medications (pl.DataFrame | pl.LazyFrame): Medication data.
        admits_last (pl.DataFrame | pl.LazyFrame): Final hospitalisations data.
        top_n (int): Number of top medications to include.
        use_lazy (bool): If True, return a LazyFrame.

    Returns:
        pl.DataFrame or pl.LazyFrame: Admissions data with medication count features.
    """
```

---

## `encode_categorical_features`

```python
def encode_categorical_features(ehr_data: pl.DataFrame) -> pl.DataFrame:
    """
    Apply one-hot encoding to categorical features in EHR data.

    Args:
        ehr_data (pl.DataFrame): Static EHR dataset.

    Returns:
        pl.DataFrame: Transformed EHR data.
    """
```

---

## `extract_lookup_fields`

```python
def extract_lookup_fields(
    ehr_data: pl.DataFrame,
    lookup_list: list = None,
    lookup_output_path: str = "../outputs/reference",
) -> pl.DataFrame:
    """
    Extract date and summary fields not suitable for training into a separate DataFrame.

    Args:
        ehr_data (pl.DataFrame): Static EHR dataset.
        lookup_list (list): List of columns to extract.
        lookup_output_path (str): Directory to save lookup fields.

    Returns:
        pl.DataFrame: EHR data with lookup fields removed.
    """
```

---

## `remove_correlated_features`

```python
def remove_correlated_features(
    ehr_data: pl.DataFrame,
    feats_to_save: list = None,
    threshold: float = 0.9,
    method: str = "pearson",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Drop highly correlated features from EHR data, keeping specified features.

    Args:
        ehr_data (pl.DataFrame): Static EHR dataset.
        feats_to_save (list): Features to keep.
        threshold (float): Correlation threshold.
        method (str): Correlation method. Defaults to Pearson's R.
        verbose (bool): If True, print summary.

    Returns:
        pl.DataFrame: EHR data with correlated features removed.
    """
```

---

## `generate_train_val_test_set`

```python
def generate_train_val_test_set(
    ehr_data: pl.DataFrame,
    output_path: str = "../outputs/processed_data",
    outcome_col: str = "in_hosp_death",
    output_summary_path: str = "../outputs/exp_data",
    seed: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    cont_cols: list = None,
    nn_cols: list = None,
    disp_dict: dict = None,
    stratify: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Create train/val/test split from static EHR data and save patient IDs across each split.

    Args:
        ehr_data (pl.DataFrame): Static EHR dataset.
        output_path (str): Directory to save split IDs.
        outcome_col (str): Outcome column name.
        output_summary_path (str): Directory to save summary.
        seed (int): Random seed.
        train_ratio (float): Proportion for training set.
        val_ratio (float): Proportion for validation set.
        test_ratio (float): Proportion for test set.
        cont_cols (list): Continuous columns.
        nn_cols (list): Non-normal columns.
        disp_dict (dict): Display name mapping.
        stratify (bool): If True, stratify splits balancing the sets by outcome prevalence, gender and ethnicity.
        verbose (bool): If True, print summary.

    Returns:
        dict: Dictionary with train, val, and test DataFrames.
    """
```

---

## `clean_notes`

```python
def clean_notes(notes: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Clean notes data by removing special characters and extra whitespaces.

    Args:
        notes (pl.DataFrame | pl.LazyFrame): Notes data.

    Returns:
        pl.DataFrame or pl.LazyFrame: Cleaned notes data.
    """
```

---

## `process_text_to_embeddings`

```python
def process_text_to_embeddings(notes: pl.DataFrame) -> dict:
    """
    Generate embeddings using the Bio+Discharge ClinicalBERT model pre-trained on MIMIC-III discharge summaries.
    The current setup uses a SpaCy tokenizer mapped to a PyTorch object for GPU support.
    Text length is limited to 128 tokens per clinical note, with included padding and truncation where appropriate.
    The pre-trained model is provided by Alsentzer et al. (https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT).

    Args:
        notes (pl.DataFrame): DataFrame containing notes data.

    Returns:
        dict: Mapping from subject_id to list of (sentence, embedding) pairs.
    """
```

---

## `clean_labevents`

```python
def clean_labevents(labs_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Clean lab events by removing non-integer values and outliers.

    Args:
        labs_data (pl.LazyFrame): Lab events data.

    Returns:
        pl.LazyFrame: Cleaned lab events.
    """
```

---

## `add_time_elapsed_to_events`

```python
def add_time_elapsed_to_events(
    events: pl.DataFrame, starttime: pl.Datetime, remove_charttime: bool = False
) -> pl.DataFrame:
    """
    Add a column for time elapsed since a reference start time.

    Args:
        events (pl.DataFrame): Events table.
        starttime (pl.Datetime): Reference start time.
        remove_charttime (bool): If True, remove charttime column.

    Returns:
        pl.DataFrame: Updated events table.
    """
```

---

## `convert_events_to_timeseries`

```python
def convert_events_to_timeseries(events: pl.DataFrame) -> pl.DataFrame:
    """
    Convert long-form events to wide-form time-series.

    Args:
        events (pl.DataFrame): Long-form events.

    Returns:
        pl.DataFrame: Wide-form time-series.
    """
```

---

## `generate_interval_dataset`

```python
def generate_interval_dataset(
    ehr_static: pl.DataFrame,
    ts_data: pl.DataFrame,
    ehr_regtime: pl.DataFrame,
    vitals_freq: str = "5h",
    lab_freq: str = "1h",
    min_events: int = None,
    max_events: int = None,
    impute: str = "value",
    include_dyn_mean: bool = False,
    no_resample: bool = False,
    standardize: bool = False,
    max_elapsed: int = None,
    vitals_lkup: list = None,
    outcomes: list = None,
    verbose: bool = True,
) -> dict:
    """
    Generate a time-series dataset with set intervals for each event source (vital signs and lab measurements).

    Args:
        ehr_static (pl.DataFrame): Static EHR data.
        ts_data (pl.DataFrame): Time-series data.
        ehr_regtime (pl.DataFrame): Lookup dataframe for ED arrival times.
        vitals_freq (str): Frequency for vitals resampling.
        lab_freq (str): Frequency for labs resampling.
        min_events (int): Include only patients with a minimum number of events.
        max_events (int): Include only patients with a maximum number of events.
        impute (str): Imputation method. Options are "value" (filling with -1), "forward" filling, "backward" filling or "mask" creating a string indicator for missingness.
        include_dyn_mean (bool): If True, add dynamic mean features to static dataset.
        no_resample (bool): If True, skip resampling.
        standardize (bool): If True, standardize data using min-max scaling.
        max_elapsed (int): Restrict collected measurements within the set hours from ED arrival.
        vitals_lkup (list): List of vital sign features.
        outcomes (list): List of outcome columns.
        verbose (bool): If True, print summary.

    Returns:
        dict: Data dictionary and column dictionary.
    """
```

---

## `_prepare_feature_map_and_freq`

```python
def _prepare_feature_map_and_freq(
    ts_data: pl.DataFrame, vitals_freq: str = "5h", lab_freq: str = "1h"
) -> tuple[dict, dict]:
    """
    Prepare a mapping of feature names and frequency for each time-series source.

    Args:
        ts_data (pl.DataFrame): Time-series data containing a 'linksto' column.
        vitals_freq (str): Frequency for vital signs.
        lab_freq (str): Frequency for lab measurements.

    Returns:
        tuple: (feature_map, freq) where feature_map is a dict mapping data source to features,
               and freq is a dict mapping data source to frequency string.
    """
```

---

## `_process_patient_events`

```python
def _process_patient_events(
    pt_events: pl.DataFrame,
    feature_map: dict,
    freq: dict,
    ehr_static: pl.DataFrame,
    edregtime: pl.Datetime,
    min_events: int = 1,
    max_events: int = None,
    impute: str = "value",
    include_dyn_mean: bool = False,
    no_resample: bool = False,
    max_elapsed: int = None,
) -> tuple[bool, list[pl.DataFrame]]:
    """
    Process time-series events for a single patient, handling missing features, imputation, resampling, and filtering.

    Args:
        pt_events (pl.DataFrame): Patient's time-series events.
        feature_map (dict): Mapping from source to feature names.
        freq (dict): Mapping from source to frequency string.
        ehr_static (pl.DataFrame): Static EHR data for the patient.
        edregtime (pl.Datetime): Lookup dataframe for ED registration time.
        min_events (int): Minimum number of measurements required.
        max_events (int): Maximum number of measurements required.
        impute (str): Imputation method. Options are "value" (filling with -1), "forward" filling, "backward" filling or "mask" creating a string indicator for missingness.
        include_dyn_mean (bool): If True, add dynamic mean features.
        no_resample (bool): If True, skip resampling.
        max_elapsed (int): Restrict collected measurements within the set hours from ED arrival.

    Returns:
        tuple: (write_data, ts_data_list, skipped_due_to_event_count, skipped_due_to_elapsed_time)
    """
```

---

## `_validate_event_count`

```python
def _validate_event_count(
    timeseries: pl.DataFrame, min_events: int = 1, max_events: int = 1e6
) -> bool:
    """
    Check if the number of events in the timeseries is within the specified range.

    Args:
        timeseries (pl.DataFrame): Time-series data.
        min_events (int): Minimum number of events.
        max_events (int): Maximum number of events.

    Returns:
        bool: True if within range, False otherwise.
    """
```

---

## `_handle_missing_features`

```python
def _handle_missing_features(
    timeseries: pl.DataFrame, features: list[str] = None
) -> pl.DataFrame:
    """
    Add missing columns to the timeseries DataFrame as nulls.

    Args:
        timeseries (pl.DataFrame): Time-series data.
        features (list): List of required feature names.

    Returns:
        pl.DataFrame: Time-series data with missing columns added as nulls.
    """
```

---

## `_impute_missing_values`

```python
def _impute_missing_values(
    timeseries: pl.DataFrame, ehr_static: pl.DataFrame, impute: str = "value"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Impute missing values in time-series and static EHR data.

    Args:
        timeseries (pl.DataFrame): Time-series data.
        ehr_static (pl.DataFrame): Static EHR data.
        impute (str): Imputation method ("mask", "forward", "backward", "value").

    Returns:
        tuple: (imputed_timeseries, imputed_ehr_static)
    """
```

---

## `_add_dynamic_mean`

```python
def _add_dynamic_mean(
    timeseries: pl.DataFrame, ehr_static: pl.DataFrame
) -> pl.DataFrame:
    """
    Add mean of dynamic features to the static EHR data.

    Args:
        timeseries (pl.DataFrame): Time-series data.
        ehr_static (pl.DataFrame): Static EHR data.

    Returns:
        pl.DataFrame: Static EHR data with dynamic means appended.
    """
```

---

## `_resample_timeseries`

```python
def _resample_timeseries(timeseries: pl.DataFrame, freq: str = "1h") -> pl.DataFrame:
    """
    Resample the time-series data to a specified frequency.

    Args:
        timeseries (pl.DataFrame): The input time-series data.
        freq (str): The frequency for resampling (e.g., "1h").

    Returns:
        pl.DataFrame: The resampled time-series data.
    """
```

---

## `_standardize_data`

```python
def _standardize_data(ts_data: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize the 'value' column in the time-series data using min-max scaling.

    Args:
        ts_data (pl.DataFrame): The input time-series data.

    Returns:
        pl.DataFrame: Standardized time-series data.
    """
```

---

## `_print_summary`

```python
def _print_summary(
    n: int = 0,
    filter_by_nb_events: int = 0,
    missing_event_src: int = 0,
    filter_by_elapsed_time: int = 0,
) -> None:
    """
    Print a summary of the time-series interval generation process.

    Args:
        n (int): Number of successfully processed patients.
        filter_by_nb_events (int): Number of patients skipped due to event count.
        missing_event_src (int): Number of patients skipped due to missing sources.
        filter_by_elapsed_time (int): Number of patients skipped due to elapsed time.

    Returns:
        None
    """
```
