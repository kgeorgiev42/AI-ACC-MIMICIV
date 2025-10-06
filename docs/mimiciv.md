# Functions for reading and processing the downloaded MIMIC-IV (v3.1) data

This document provides an overview of the functions defined in `src.utils.mimiciv`. Each function is listed with its signature and docstring.

---

## `read_admissions_table`

```python
def read_admissions_table(
    mimic4_path: str,
    use_lazy: bool = False,
    verbose: bool = True,
    ext_stay_threshold: int = 7,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the admissions table from MIMIC-IV, setting up the ED population.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV hospital module files.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        verbose (bool): If True, print summary statistics.
        ext_stay_threshold (int): Threshold (in days) for setting extended stay outcome.

    Returns:
        pl.LazyFrame | pl.DataFrame: Admissions table with additional columns.
    """
```

---

## `read_patients_table`

```python
def read_patients_table(
    mimic4_path: str,
    admissions_data: pl.DataFrame | pl.LazyFrame,
    age_cutoff: int = 18,
    use_lazy: bool = False,
    verbose: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the patients table from MIMIC-IV and join with admissions.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admissions_data (pl.DataFrame | pl.LazyFrame): Admissions table.
        age_cutoff (int): Minimum age to include.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        verbose (bool): If True, print summary statistics.

    Returns:
        pl.LazyFrame | pl.DataFrame: Patients table with joined admissions and derived outcomes.
    """
```

---

## `read_icu_table`

```python
def read_icu_table(
    mimic4_ed_path: str,
    admissions_data: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
    verbose: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the ICU stays table and join with admissions.

    Args:
        mimic4_ed_path (str): Path to directory containing MIMIC-IV module files.
        admissions_data (pl.DataFrame | pl.LazyFrame): Admissions table.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        verbose (bool): If True, print summary statistics.

    Returns:
        pl.LazyFrame | pl.DataFrame: ICU stays table with joined admissions and derived columns.
    """
```

---

## `read_d_icd_diagnoses_table`

```python
def read_d_icd_diagnoses_table(mimic4_path):
    """
    Read the ICD diagnoses dictionary table from MIMIC-IV.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.

    Returns:
        pl.DataFrame: ICD diagnoses dictionary table.
    """
```

---

## `read_diagnoses_table`

```python
def read_diagnoses_table(
    mimic4_path: str,
    admissions_data: pl.DataFrame | pl.LazyFrame,
    adm_last: pl.DataFrame | pl.LazyFrame,
    verbose: bool = True,
    use_lazy: bool = False,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the diagnoses table from MIMIC-IV and join with admissions.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admissions_data (pl.DataFrame | pl.LazyFrame): Admissions table.
        adm_last (pl.DataFrame | pl.LazyFrame): Final hospitalisations table for looking up prior diagnoses.
        verbose (bool): If True, print summary statistics.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.

    Returns:
        pl.LazyFrame | pl.DataFrame: Diagnoses table filtered and joined with admissions.
    """
```

---

## `read_notes`

```python
def read_notes(
    admissions_data: pl.DataFrame | pl.LazyFrame,
    admits_last: pl.DataFrame | pl.LazyFrame,
    mimic4_path: str,
    verbose: bool = True,
    use_lazy: bool = False,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess discharge summary and link Brief Hospital Course segments.

    Args:
        admissions_data (pl.DataFrame | pl.LazyFrame): Admissions table.
        admits_last (pl.DataFrame | pl.LazyFrame): Final hospitalisations table for looking up notes history.
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        verbose (bool): If True, print summary statistics.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.

    Returns:
        pl.LazyFrame | pl.DataFrame: Notes table joined with admissions and BHC segments.
    """
```

---

## `get_notes_population`

```python
def get_notes_population(
    adm_notes: pl.DataFrame | pl.LazyFrame,
    admit_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
) -> pl.DataFrame:
    """
    Get population of unique ED patients with existing note history.

    Args:
        adm_notes (pl.DataFrame | pl.LazyFrame): Notes table.
        admit_last (pl.DataFrame | pl.LazyFrame): Last hospitalisations table for looking up notes history.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.

    Returns:
        tuple: Patients and Grouped notes table (ed_pts, notes_grouped) as DataFrames or LazyFrames.
    """
```

---

## `read_omr_table`

```python
def read_omr_table(
    mimic4_path: str,
    admits_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
    vitalsign_uom_map: dict = None,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the OMR table from MIMIC-IV and join with admissions.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admits_last (pl.DataFrame | pl.LazyFrame): Last hospitalisations table.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        vitalsign_uom_map (dict): Optional mapping of vital sign units of measure.

    Returns:
        pl.LazyFrame | pl.DataFrame: OMR table with joined admissions and processed vital signs.
    """
```

---

## `read_vitals_table`

```python
def read_vitals_table(
    mimic4_ed_path: str,
    admits_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
    vitalsign_column_map: dict = None,
    vitalsign_uom_map: dict = None,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the vitals table from MIMIC-IV and join with admissions.

    Args:
        mimic4_ed_path (str): Path to directory containing MIMIC-IV module files.
        admits_last (pl.DataFrame | pl.LazyFrame): Last hospitalisations table.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        vitalsign_column_map (dict): Optional mapping of vitals column names.
        vitalsign_uom_map (dict): Optional mapping of vital sign units of measure.

    Returns:
        pl.LazyFrame | pl.DataFrame: Vitals table with joined admissions and processed columns.
    """
```

---

## `read_labevents_table`

```python
def read_labevents_table(
    mimic4_path: str,
    admits_last: pl.DataFrame | pl.LazyFrame,
    include_items: str = "../config/lab_items.csv",
) -> pl.LazyFrame:
    """
    Read and preprocess the lab events table from MIMIC-IV and join with admissions.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admits_last (pl.DataFrame | pl.LazyFrame): Last hospitalisations table.
        include_items (str): Path to CSV file with items to include.

    Returns:
        pl.LazyFrame: Lab events table with joined admissions and filtered items.
    """
```

---

## `merge_events_table`

```python
def merge_events_table(
    vitals: pl.LazyFrame | pl.DataFrame,
    labs: pl.LazyFrame | pl.DataFrame,
    omr: pl.LazyFrame | pl.DataFrame,
    use_lazy: bool = False,
    verbose: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Merge vitals, labs, and OMR events into a single table.

    Args:
        vitals (pl.LazyFrame | pl.DataFrame): Vitals table.
        labs (pl.LazyFrame | pl.DataFrame): Labs table.
        omr (pl.LazyFrame | pl.DataFrame): OMR table.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        verbose (bool): If True, print summary statistics.

    Returns:
        pl.LazyFrame | pl.DataFrame: Merged events table with additional columns.
    """
```

---

## `get_population_with_measures`

```python
def get_population_with_measures(
    events: pl.DataFrame | pl.LazyFrame,
    admit_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
) -> pl.DataFrame:
    """
    Get population of patients with available measurements.

    Args:
        events (pl.DataFrame | pl.LazyFrame): Events table (vitals, labs, omr).
        admit_last (pl.DataFrame | pl.LazyFrame): Last hospitalisations table.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.

    Returns:
        pl.DataFrame: Filtered population with available measurements.
    """
```

---

## `read_medications_table`

```python
def read_medications_table(
    mimic4_path: str,
    admits_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
    top_n: int = 50,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the medications table from MIMIC-IV and join with admissions.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admits_last (pl.DataFrame | pl.LazyFrame): Last hospitalisations table.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        top_n (int): Number of top medications to include.

    Returns:
        pl.LazyFrame | pl.DataFrame: Medications table with joined admissions and top N medications.
    """
```

---

## `read_specialty_table`

```python
def read_specialty_table(
    mimic4_path: str, admits_last: pl.DataFrame | pl.LazyFrame, use_lazy: bool = False
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the specialty table from MIMIC-IV and join with admissions.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admits_last (pl.DataFrame | pl.LazyFrame): Last hospitalisations table.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.

    Returns:
        pl.LazyFrame | pl.DataFrame: Specialty table with joined admissions and derived columns.
    """
```

---

## `save_multimodal_dataset`

```python
def save_multimodal_dataset(
    admits_last: pl.DataFrame | pl.LazyFrame,
    events: pl.DataFrame | pl.LazyFrame,
    notes: pl.DataFrame | pl.LazyFrame,
    use_events: bool = True,
    use_notes: bool = True,
    output_path: str = "../outputs/extracted_data",
):
    """
    Save the multimodal dataset to disk.

    Args:
        admits_last (pl.DataFrame | pl.LazyFrame): Last hospitalisations table.
        events (pl.DataFrame | pl.LazyFrame): Events table (vitals, labs, omr).
        notes (pl.DataFrame | pl.LazyFrame): Notes table.
        use_events (bool): If True, include events in the output.
        use_notes (bool): If True, include notes in the output.
        output_path (str): Directory to save the output files.

    Returns:
        None
    """
```
