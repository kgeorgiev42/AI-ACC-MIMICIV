import json
import os

import numpy as np
import pandas as pd
import polars as pl
from utils.preprocessing import (
    clean_labevents,
    prepare_medication_features,
    transform_sensitive_attributes,
)
from utils.functions import get_n_unique_values
import re


def read_admissions_table(
    mimic4_path: str,
    mimic4_ed_path: str,
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
    admits = pl.read_csv(
        os.path.join(mimic4_path, "admissions.csv.gz"),
        columns=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "edregtime",
            "edouttime",
            "insurance",
            "marital_status",
            "race",
            "admission_location",
            "discharge_location",
        ],
        dtypes=[
            pl.Int64,
            pl.Int64,
            pl.Datetime,
            pl.Datetime,
            pl.Datetime,
            pl.Datetime,
            pl.Datetime,
            pl.String,
            pl.String,
            pl.String,
            pl.String,
            pl.String,
        ],
        try_parse_dates=True,
    )

    admits = admits.filter(
        pl.col("edregtime").is_not_null()
        & pl.col("edouttime").is_not_null()
        & pl.col("marital_status").is_not_null()
        & pl.col("race").is_not_null()
        & pl.col("insurance").is_not_null()
    )
    if verbose:
        print(
            "Number of admissions with complete data:",
            admits.shape[0],
            admits.select("subject_id").n_unique(),
        )

    admits = admits.filter(
        (pl.col("edregtime") < pl.col("admittime"))
        & (pl.col("edregtime") < pl.col("dischtime"))
        & (pl.col("edouttime") < pl.col("dischtime"))
        & (pl.col("edouttime") > pl.col("edregtime"))
        & (pl.col("admittime") < pl.col("dischtime"))
    )
    if verbose:
        print(
            "Number of admissions after timestamp validation:",
            admits.shape[0],
            admits.select("subject_id").n_unique(),
        )

    admits = admits.with_columns(
        (
            (pl.col("dischtime") - pl.col("admittime")).dt.seconds() / (24 * 60 * 60)
        ).alias("los_days")
    )

    admits = admits.with_columns(
        (pl.col("los_days") > ext_stay_threshold).cast(pl.Int8).alias("ext_stay_7")
    )

    ### Get ED attendance metadata
    ed_stays = pl.read_csv(
        os.path.join(mimic4_ed_path, "edstays.csv.gz"),
        columns=['subject_id', 'hadm_id', 'stay_id', 'intime', 'arrival_transport', 'disposition'],
        dtypes=[pl.Int64, pl.Int64, pl.Int64, pl.Datetime, pl.String, pl.String],
        try_parse_dates=True,
    )
    ed_stays = ed_stays.filter(pl.col("disposition") == "ADMITTED")
    ed_stays = ed_stays.filter(pl.col("hadm_id").is_in(admits.select("hadm_id")))
    ed_stays = ed_stays.sort(by=["subject_id", "hadm_id", "intime"])
    ed_stays = ed_stays.unique(subset=["hadm_id"], keep="last")
    ed_stays = ed_stays.rename({"stay_id": "ed_stay_id", "intime": "ed_intime"})
    ## Recode arrival modes
    ed_stays = ed_stays.with_columns(
        pl.when(pl.col('arrival_transport').is_in(['HELICOPTER', 'OTHER']))
        .then(pl.lit('AMBULANCE'))
        .otherwise(pl.col('arrival_transport'))
        .alias('arrival_transport')
    )
    admits = admits.join(ed_stays, on=["subject_id", "hadm_id"], how="left")
    admits = admits.filter(pl.col("ed_stay_id").is_not_null())
    admits = admits.with_columns(pl.col("ed_stay_id").cast(pl.Int64))

    if verbose:
        print(
            f'Subjects with extended stay > 7 days: {admits.filter(pl.col("ext_stay_7") == 1).select("subject_id").n_unique()}, % of pts: {admits.filter(pl.col("ext_stay_7") == 1).select("subject_id").n_unique() / admits.select("subject_id").n_unique() * 100:.2f}'
        )

    print("Collected admissions table and linked ED attendances..")
    return admits.lazy() if use_lazy else admits


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
    pats = pl.read_csv(
        os.path.join(mimic4_path, "patients.csv.gz"),
        columns=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
        dtypes=[pl.Int64, pl.String, pl.Int64, pl.Int64, pl.Datetime],
        try_parse_dates=True,
    )

    if isinstance(admissions_data, pl.LazyFrame):
        admissions_data = admissions_data.collect()

    pats = pats.filter(
        pl.col("subject_id").is_in(admissions_data.select("subject_id").to_series())
    )
    pats = pats.select(["subject_id", "gender", "dod", "anchor_age", "anchor_year"])
    pats = pats.with_columns(
        (pl.col("anchor_year") - pl.col("anchor_age")).alias("yob")
    ).drop("anchor_year")
    pats = pats.join(admissions_data, on="subject_id", how="left")
    pats = pats.filter(pl.col("anchor_age") >= age_cutoff)
    pats = pats.with_columns(pl.col("discharge_location").fill_null("UNKNOWN"))
    pats = pats.with_columns(
        ((pl.col("dod") <= pl.col("dischtime")) & (pl.col("dod") > pl.col("admittime")))
        .cast(pl.Int8)
        .alias("in_hosp_death")
    )
    pats = pats.with_columns(
        (
            (
                ~pl.col("discharge_location").is_in(
                    [
                        "HOME",
                        "HOME HEALTH CARE",
                        "DIED",
                        "AGAINST ADVICE",
                        "ASSISTED LIVING",
                        "UNKNOWN",
                    ]
                )
            )
            & (pl.col("in_hosp_death") == 0)
        )
        .cast(pl.Int8)
        .alias("non_home_discharge")
    )
    pats = pats.with_columns(
        pl.col("in_hosp_death").fill_null(0).cast(pl.Int8),
        pl.col("non_home_discharge").fill_null(0).cast(pl.Int8),
    )
    pats = transform_sensitive_attributes(pats)

    if verbose:
        print(
            f"Subjects with in-hospital death: {pats.filter(pl.col('in_hosp_death') == 1).select('subject_id').n_unique()}, % of pts: {pats.filter(pl.col('in_hosp_death') == 1).select('subject_id').n_unique() / pats.select('subject_id').n_unique() * 100:.2f}"
        )
        print(
            f"Subjects with non-home discharge: {pats.filter(pl.col('non_home_discharge') == 1).select('subject_id').n_unique()}, % of pts: {pats.filter(pl.col('non_home_discharge') == 1).select('subject_id').n_unique() / pats.select('subject_id').n_unique() * 100:.2f}"
        )
    print("Collected patients table linked to ED attendances..")
    return pats.lazy() if use_lazy else pats


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
    icu = pl.read_csv(
        os.path.join(mimic4_ed_path, "icustays.csv.gz"),
        columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        dtypes=[pl.Int64, pl.Int64, pl.Int64, pl.Datetime, pl.Datetime, pl.Float32],
        try_parse_dates=True,
    )

    if isinstance(admissions_data, pl.LazyFrame):
        admissions_data = admissions_data.collect()

    if verbose:
        print(
            "Original number of ICU stays:",
            icu.shape[0],
            icu.select("subject_id").n_unique(),
        )
    icu = icu.filter(
        pl.col("subject_id").is_in(admissions_data.select("subject_id"))
        & pl.col("hadm_id").is_in(admissions_data.select("hadm_id"))
    )
    if verbose:
        print(
            "Number of ICU stays with validated ED attendances:",
            icu.shape[0],
            icu.select("subject_id").n_unique(),
        )
    icu_eps = (
        admissions_data.join(icu, on=["subject_id", "hadm_id"], how="left")
        .sort(by=["subject_id", "hadm_id", "intime"])
        .unique(subset=["subject_id", "hadm_id"], keep="last")
    )

    icu_eps = icu_eps.with_columns(
        (
            (pl.col("intime") > pl.col("admittime"))
            & (pl.col("outtime") < pl.col("dischtime"))
        )
        .cast(pl.Int8)
        .alias("icu_admission"),
        pl.col("los").alias("icu_los_days"),
    )
    icu_eps = icu_eps.with_columns(
        pl.col("icu_admission").fill_null(0).cast(pl.Int8),
        pl.col("icu_los_days").fill_null(0).cast(pl.Int8),
    )
    icu_eps = icu_eps.drop(["los"])
    if verbose:
        print(
            f'Subjects with ICU admission: {icu_eps.filter(pl.col("icu_admission") == 1).select("subject_id").n_unique()}, % of pts: {icu_eps.filter(pl.col("icu_admission") == 1).select("subject_id").n_unique() / icu_eps.select("subject_id").n_unique() * 100:.2f}'
        )
    print("Collected ICU stay outcomes..")
    return icu_eps.lazy() if use_lazy else icu_eps

### ECG detection patterns ###
ACUTE_MI_PATTERN = re.compile(
    r'\b('
    r'(acute\s+(myocardial\s+)?infarction)'      # acute MI or acute myocardial infarction
    r'|(st[-\s]*elevation\s+mi)'                 # STEMI full form
    r'|(non[-\s]*st[-\s]*elevation\s+mi)'        # NSTEMI full form
    r'|(\bstemi\b)'                              # STEMI abbreviation
    r'|(\bnstemi\b)'                             # NSTEMI abbreviation
    r'|(q[-\s]*wave\s+mi)'                       # older term Q-wave MI
    r'|(subendocardial\s+(mi|infarction))'       # subendocardial MI (NSTEMI equivalent)
    r')\b',
    flags=re.IGNORECASE
)

ISCHEMIA_PATTERN = re.compile(
    r'\b('
    r'(st[-\s]?segment\s?(depression|elevation|changes|shift))|'  # ST-segment changes
    r'(t[-\s]?wave\s?(inversion|flattening|abnormality))|'        # T-wave issues
    r'(q[-\s]?wave\s?(abnormality|pathological))|'                # Pathological Q waves
    r'((subendocardial|transmural)\s?ischemi(a|c))|'              # Variants of ischemia
    r'(\bischemi(a|c)\b)|'                                        # Generic ischemia mentions
    r'(unstable\sangina)|'                                        # Unstable angina pattern
    r'(nste(\s|-)?acs|nstemi)'                                    # NSTEMI or NSTE-ACS
    r')\b',
    flags=re.IGNORECASE
)

ST_ELEVATION_PATTERN = re.compile(r"""
\b(
    ST[\s\-]?elev(?:ation|ated|ating|s)? |      # “ST elevation”, “ST-elevated”, etc.
    ST[\s\-]?segment[\s\-]elev(?:ation|ated|s)? |  # “ST-segment elevation”
    STEMI |                                       # Explicit STEMI mention
    (acute\s+myocardial\s+infarction) |          # Phrases like “acute myocardial infarction”
    (elevation\s+in\s+[Vv]\d+(-[Vv]\d+)?) |      # e.g., “elevation in V2-V4”
    (marked\s+ST\s+changes?) |                   # “marked ST changes”
    (concave\s+ST\s+segments?) |                 # morphological variant
    (convex\s+ST\s+segments?)                    # morphological variant
)\b
""", flags=re.IGNORECASE | re.VERBOSE)

def read_ecg_measurements(
    mimic4_ecg_path: str,
    mimic4_ed_path: str,
    admissions_data: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
    verbose: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the ECG measurements table from MIMIC-IV and join with admissions.

    Args:
        mimic4_ecg_path (str): Path to directory containing MIMIC-IV module files.
        admissions_data (pl.DataFrame | pl.LazyFrame): Admissions table.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        verbose (bool): If True, print summary statistics.

    Returns:
        pl.LazyFrame | pl.DataFrame: ECG measurements table filtered and joined with admissions.
    """
    ecg_measures = pl.read_csv(
        os.path.join(mimic4_ecg_path, "machine_measurements.csv"),
        columns=["subject_id", "study_id", "cart_id", "ecg_time", "report_0",
                 "report_1", "report_2", "report_3", "report_4", "report_5", "report_6",
                 "report_7", "report_8", "report_9", "report_10", "report_11", "report_12",
                 "report_13", "report_14", "report_15", "report_16", "report_17", "rr_interval",
                 "p_onset", "p_end", "qrs_onset", "qrs_end", "t_end", "p_axis", "qrs_axis", "t_axis"],
        dtypes=[pl.Int64, pl.Int64, pl.Int64, pl.Datetime, pl.String, pl.String,
                pl.String, pl.String, pl.String, pl.String, pl.String, pl.String,
                pl.String, pl.String, pl.String, pl.String, pl.String, pl.String,
                pl.String, pl.String, pl.String, pl.String, pl.Int64,
                pl.Int64, pl.Int64, pl.Int64, pl.Int64, pl.Int64, pl.Int64, pl.Int64, pl.Int64],
        try_parse_dates=True,
    )

    edmd_measures = pl.read_csv(
        os.path.join(mimic4_ed_path, "triage.csv.gz"),
        columns=["subject_id", "stay_id", "chiefcomplaint"],
        dtypes=[pl.Int64, pl.Int64, pl.String],
    )
    edmd_measures = edmd_measures.rename({"stay_id": "ed_stay_id"})

    if isinstance(admissions_data, pl.LazyFrame):
        admissions_data = admissions_data.collect()

    ecg_measures = ecg_measures.filter(
        pl.col("subject_id").is_in(admissions_data.select("subject_id"))
    )

    print("Collected ECG measurements table..")

    if isinstance(admissions_data, pl.LazyFrame):
        admissions_data = admissions_data.collect()

    ### ECG measurements preprocessing ###
    # Convert rr_interval to numeric first (it might be a string)
    ecg_measures = ecg_measures.with_columns(
        pl.col("rr_interval").cast(pl.Float64, strict=False)
    )

    # Estimate additional ECG measures using Polars syntax
    ecg_measures = ecg_measures.with_columns([
        (pl.col("rr_interval") / 1000).alias("rr_interval_seconds"),
        (pl.col("qrs_onset") - pl.col("p_onset")).alias("pr_interval"),
        (pl.col("p_end") - pl.col("p_onset")).alias("p_wave_duration"),
        (pl.col("qrs_end") - pl.col("qrs_onset")).alias("qrs_duration"),
        (pl.col("t_end") - pl.col("qrs_onset")).alias("qt_interval")
    ])

    # Calculate ventricular rate after rr_interval_seconds is created
    ecg_measures = ecg_measures.with_columns(
        (60 / pl.col("rr_interval_seconds")).alias("ventricular_rate")
    )

    # Check for plausibility of values using Polars when/then syntax
    ecg_measures = ecg_measures.with_columns([
        pl.when((pl.col("pr_interval") < 20) | (pl.col("pr_interval") > 500))
        .then(None)
        .otherwise(pl.col("pr_interval"))
        .alias("pr_interval"),

        pl.when((pl.col("p_wave_duration") <= 20) | (pl.col("p_wave_duration") > 300))
        .then(None)
        .otherwise(pl.col("p_wave_duration"))
        .alias("p_wave_duration"),

        pl.when((pl.col("qrs_duration") < 50) | (pl.col("qrs_duration") > 300))
        .then(None)
        .otherwise(pl.col("qrs_duration"))
        .alias("qrs_duration"),

        pl.when((pl.col("qt_interval") < 200) | (pl.col("qt_interval") > 700))
        .then(None)
        .otherwise(pl.col("qt_interval"))
        .alias("qt_interval")
    ])

    # Calculate qtc_interval using Bazett's formula
    ecg_measures = ecg_measures.with_columns(
        (pl.col("qt_interval") / (60 / pl.col("ventricular_rate")).sqrt()).alias("qtc_interval")
    )

    # Drop the temporary rr_interval_seconds column
    ecg_measures = ecg_measures.drop("rr_interval_seconds")

    # Replace axis values with None if corresponding duration/interval is null or out of range
    ecg_measures = ecg_measures.with_columns([
        pl.when(pl.col("p_wave_duration").is_null())
        .then(None)
        .when((pl.col("p_axis") < -90) | (pl.col("p_axis") > 270))
        .then(None)
        .otherwise(pl.col("p_axis"))
        .alias("p_axis"),

        pl.when(pl.col("qrs_duration").is_null())
        .then(None)
        .when((pl.col("qrs_axis") < -90) | (pl.col("qrs_axis") > 270))
        .then(None)
        .otherwise(pl.col("qrs_axis"))
        .alias("qrs_axis"),

        pl.when(pl.col("qt_interval").is_null())
        .then(None)
        .when((pl.col("t_axis") < -90) | (pl.col("t_axis") > 270))
        .then(None)
        .otherwise(pl.col("t_axis"))
        .alias("t_axis")
    ])

    #ecg_measures = ecg_measures.with_columns([pl.col(col).fill_null(-1) for col in ['p_axis', 'qrs_axis', 't_axis', 'pr_interval', 'p_wave_duration', 'qrs_duration', 'qt_interval', 'ventricular_rate', 'qtc_interval']])

    ## Diagnostics
    # Create full_report by merging all report columns (report_0 to report_17)
    report_cols = [f'report_{i}' for i in range(18)]  # report_0 to report_17
    existing_report_cols = [col for col in report_cols if col in ecg_measures.columns]

    print(f"Found report columns: {existing_report_cols}")

    # Fill null values with empty string and concatenate using Polars
    ecg_measures = ecg_measures.with_columns([
        pl.col(col).fill_null("").alias(col) for col in existing_report_cols
    ])

    # Concatenate all report columns with space separator
    ecg_measures = ecg_measures.with_columns(
        pl.concat_str(existing_report_cols, separator=" ").alias("full_report")
    )

    # Clean up the full_report text using Polars string methods
    ecg_measures = ecg_measures.with_columns(
        pl.col("full_report")
        .str.replace_all("\n", " ")
        .str.replace_all("\r", " ")
        .str.replace_all("  ", " ")
        .str.strip_chars()
        .str.to_lowercase()
        .alias("full_report")
    )

    # Create clinical indicators from the full report using Polars
    ecg_measures = ecg_measures.with_columns([
        pl.when(
            pl.col("full_report").is_not_null() &
            (pl.col("full_report").str.contains("sinus rhythm") |
             pl.col("full_report").str.contains("sinus tachycardia") |
             pl.col("full_report").str.contains("sinus bradycardia")) &
            ~pl.col("full_report").str.contains("abnormal ecg")
        )
        .then(1)
        .otherwise(0)
        .alias("ecg_normal")
    ])

    # Use regex patterns to identify ST-elevation and myocardial ischemia
    # Use compiled regex patterns from module level
    acute_mi_pattern = ACUTE_MI_PATTERN
    ischemia_pattern = ISCHEMIA_PATTERN
    st_elevation_pattern = ST_ELEVATION_PATTERN

    # Helper function for indicator columns using Python regex search
    def add_ecg_indicator(df, col_name, pattern):
        return df.with_columns(
            pl.col("full_report")
            .apply(lambda s: 1 if (s is not None and pattern.search(s)) else 0)
            .cast(pl.Int8)
            .alias(col_name)
        )

    ecg_measures = add_ecg_indicator(ecg_measures, "ecg_st_elevation", st_elevation_pattern)
    ecg_measures = add_ecg_indicator(ecg_measures, "ecg_myocardial_ischemia", ischemia_pattern)
    ecg_measures = add_ecg_indicator(ecg_measures, "ecg_acute_mi", acute_mi_pattern)

    # Merge with admissions data to filter relevant ECGs
    ecg_admits = admissions_data.join(ecg_measures.select(["subject_id", "ecg_time", "full_report", "ecg_normal", "ecg_st_elevation", "ecg_myocardial_ischemia", "ecg_acute_mi",
                                                           "ventricular_rate", "pr_interval", "qrs_duration", "qt_interval", "qtc_interval",
                                                           "p_axis", "qrs_axis", "t_axis"]), on="subject_id", how="left")

    # Filter the ECGs to keep only those within 3 hours after ED presentation using Polars
    ecg_admits = ecg_admits.filter(
        (pl.col("ecg_time") >= pl.col("edregtime")) &
        (pl.col("ecg_time") <= pl.col("edregtime") + pl.duration(hours=3))
    )
    ecg_admits = ecg_admits.sort(by=["subject_id", "hadm_id", "ecg_time"]).unique(subset=["subject_id", "hadm_id"], keep="last")

    # Get presenting complaint
    edmd_measures = edmd_measures.filter(
        pl.col("subject_id").is_in(ecg_admits.select("subject_id")) &
        pl.col("ed_stay_id").is_in(ecg_admits.select("ed_stay_id"))
    )

    # Define regex patterns to capture variations of chest pain, dyspnea, and palpitations
    symptoms_pattern = r'(?i)(chest[\s-]*pain|dyspnea|dysnea|shortness[\s-]*of[\s-]*breath|palpitation|palpitations|chest[\s-]*tightness|angina pectoris)'
    chest_pain_pattern = r'(?i)(chest[\s-]*pain|chest[\s-]*tightness|angina pectoris)'
    dyspnea_pattern = r'(?i)(dyspnea|dysnea|shortness[\s-]*of[\s-]*breath)'
    palpitations_pattern = r'(?i)(palpitation|palpitations)'

    # Filter for each symptom using Polars string contains
    stay_ids_symp = edmd_measures.filter(
        pl.col("chiefcomplaint").str.contains(symptoms_pattern)
    ).select("ed_stay_id").unique()

    stay_ids_chest = edmd_measures.filter(
        pl.col("chiefcomplaint").str.contains(chest_pain_pattern)
    ).select("ed_stay_id").unique()

    stay_ids_dysp = edmd_measures.filter(
        pl.col("chiefcomplaint").str.contains(dyspnea_pattern)
    ).select("ed_stay_id").unique()

    stay_ids_palp = edmd_measures.filter(
        pl.col("chiefcomplaint").str.contains(palpitations_pattern)
    ).select("ed_stay_id").unique()

    # Create symptom indicator columns using Polars
    ecg_admits = ecg_admits.with_columns([
        pl.col("ed_stay_id").is_in(stay_ids_symp).cast(pl.Int8).alias("suggestive_symptoms"),
        pl.col("ed_stay_id").is_in(stay_ids_chest).cast(pl.Int8).alias("chest_pain"),
        pl.col("ed_stay_id").is_in(stay_ids_dysp).cast(pl.Int8).alias("breathlessness"),
        pl.col("ed_stay_id").is_in(stay_ids_palp).cast(pl.Int8).alias("heart_palpitations")
    ])
    ecg_admits = ecg_admits.with_columns([pl.col(col).fill_null(0).cast(pl.Int8) for col in ['suggestive_symptoms', 'chest_pain', 'breathlessness', 'heart_palpitations']])

    if verbose:
        print(f'Number of attendances with suggestive symptoms: {ecg_admits.filter(pl.col("suggestive_symptoms") == 1).select("subject_id").n_unique()}, % of pts: {ecg_admits.filter(pl.col("suggestive_symptoms") == 1).select("subject_id").n_unique() / ecg_admits.select("subject_id").n_unique() * 100:.2f}')
        print(f'Number of attendances with chest pain: {ecg_admits.filter(pl.col("chest_pain") == 1).select("subject_id").n_unique()}, % of pts: {ecg_admits.filter(pl.col("chest_pain") == 1).select("subject_id").n_unique() / ecg_admits.select("subject_id").n_unique() * 100:.2f}')
        print(f'Number of attendances with breathlessness: {ecg_admits.filter(pl.col("breathlessness") == 1).select("subject_id").n_unique()}, % of pts: {ecg_admits.filter(pl.col("breathlessness") == 1).select("subject_id").n_unique() / ecg_admits.select("subject_id").n_unique() * 100:.2f}')
        print(f'Number of attendances with palpitations: {ecg_admits.filter(pl.col("heart_palpitations") == 1).select("subject_id").n_unique()}, % of pts: {ecg_admits.filter(pl.col("heart_palpitations") == 1).select("subject_id").n_unique() / ecg_admits.select("subject_id").n_unique() * 100:.2f}')
        print(f'Number of attendances with normal ECG: {ecg_admits.filter(pl.col("ecg_normal") == 1).select("subject_id").n_unique()}, % of pts: {ecg_admits.filter(pl.col("ecg_normal") == 1).select("subject_id").n_unique() / ecg_admits.select("subject_id").n_unique() * 100:.2f}')
        print(f'Number of attendances with ST-elevation: {ecg_admits.filter(pl.col("ecg_st_elevation") == 1).select("subject_id").n_unique()}, % of pts: {ecg_admits.filter(pl.col("ecg_st_elevation") == 1).select("subject_id").n_unique() / ecg_admits.select("subject_id").n_unique() * 100:.2f}')
        print(f'Number of attendances with myocardial ischemia: {ecg_admits.filter(pl.col("ecg_myocardial_ischemia") == 1).select("subject_id").n_unique()}, % of pts: {ecg_admits.filter(pl.col("ecg_myocardial_ischemia") == 1).select("subject_id").n_unique() / ecg_admits.select("subject_id").n_unique() * 100:.2f}')
        print(f'Number of attendances with suspected MI: {ecg_admits.filter(pl.col("ecg_acute_mi") == 1).select("subject_id").n_unique()}, % of pts: {ecg_admits.filter(pl.col("ecg_acute_mi") == 1).select("subject_id").n_unique() / ecg_admits.select("subject_id").n_unique() * 100:.2f}')

    print("Collected ECG measurements table..")
    return ecg_admits.lazy() if use_lazy else ecg_admits

def read_d_icd_diagnoses_table(mimic4_path):
    """
    Read the ICD diagnoses dictionary table from MIMIC-IV.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.

    Returns:
        pl.DataFrame: ICD diagnoses dictionary table.
    """
    d_icd = pl.read_csv(
        os.path.join(mimic4_path, "d_icd_diagnoses.csv.gz"),
        columns=["icd_code", "long_title"],
        dtypes=[pl.String, pl.String],
    )
    return d_icd


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
    diag = pl.read_csv(
        os.path.join(mimic4_path, "diagnoses_icd.csv.gz"),
        columns=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        dtypes=[pl.Int64, pl.Int64, pl.Int16, pl.String, pl.Int16],
    )
    diag_mapping = read_d_icd_diagnoses_table(mimic4_path)
    if isinstance(admissions_data, pl.LazyFrame):
        admissions_data = admissions_data.collect()
    if isinstance(adm_last, pl.LazyFrame):
        adm_last = adm_last.collect()

    diag = diag.join(diag_mapping, on="icd_code", how="inner")
    if verbose:
        print("Original number of diagnoses:", len(diag))

    # Get list of eligible hospital episodes as historical data
    adm_lkup = admissions_data.join(
        adm_last.select(["subject_id", "edregtime"]).rename(
            {"edregtime": "last_edregtime"}
        ),
        on="subject_id",
        how="left",
    )
    adm_lkup = adm_lkup.filter(pl.col("edregtime") < pl.col("last_edregtime"))

    print("Collected diagnoses table..")
    return diag.lazy() if use_lazy else diag

def read_procedures_table(
    mimic4_path: str,
    admissions_data: pl.DataFrame | pl.LazyFrame,
    adm_last: pl.DataFrame | pl.LazyFrame,
    proc_dict_path: str = None,
    verbose: bool = True,
    use_lazy: bool = False,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the procedures table from MIMIC-IV and join with admissions.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admissions_data (pl.DataFrame | pl.LazyFrame): Admissions table.
        adm_last (pl.DataFrame | pl.LazyFrame): Final hospitalisations table for looking up prior diagnoses.
        verbose (bool): If True, print summary statistics.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.

    Returns:
        pl.LazyFrame | pl.DataFrame: Procedures table filtered and joined with admissions.
    """
    proc = pl.read_csv(
        os.path.join(mimic4_path, "procedures_icd.csv.gz"),
        columns=["subject_id", "hadm_id", "icd_code"],
        dtypes=[pl.Int64, pl.Int64, pl.String],
    )

    if isinstance(admissions_data, pl.LazyFrame):
        admissions_data = admissions_data.collect()
    if isinstance(adm_last, pl.LazyFrame):
        adm_last = adm_last.collect()

    # Get list of eligible hospital episodes as historical data
    adm_lkup = admissions_data.join(
        adm_last.select(["subject_id", "edregtime"]).rename(
            {"edregtime": "last_edregtime"}
        ),
        on="subject_id",
        how="left",
    )
    adm_lkup = adm_lkup.filter(pl.col("edregtime") < pl.col("last_edregtime"))
    # Filter procedures for lookup episodes
    proc = proc.filter(pl.col("subject_id").is_in(adm_lkup.select("subject_id")))
    proc = proc.filter(pl.col("hadm_id").is_in(adm_lkup.select("hadm_id")))
    print(proc.shape, get_n_unique_values(proc, "hadm_id"))
    ### If dict is populated generate categorical columns for each procedure
    if proc_dict_path:
        if verbose:
            print(f"Loading procedure dictionary from {proc_dict_path}..")
        with open(proc_dict_path) as json_dict:
            proc_dict = json.load(json_dict)

        pci_list = proc_dict.get("proc_PCI", [])
        cag_list = proc_dict.get("proc_CAG", [])
        pci_pat = "|".join(re.escape(x) for x in pci_list) if pci_list else r"^$"
        cag_pat = "|".join(re.escape(x) for x in cag_list) if cag_list else r"^$"

        proc = proc.with_columns([
            pl.when(pl.col("icd_code").is_not_null() & pl.col("icd_code").str.contains(pci_pat))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("proc_PCI"),

            pl.when(pl.col("icd_code").is_not_null() & pl.col("icd_code").str.contains(cag_pat))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("proc_CAG")
        ])

        adm_last = adm_last.join(
            proc.select(['subject_id', 'proc_PCI', 'proc_CAG']),
            on=['subject_id'],
            how='left'
        )
        adm_last = adm_last.with_columns(
            ((pl.col('proc_PCI') == 1) | (pl.col('proc_CAG') == 1)).cast(pl.Int8).alias('prev_revasc')
        )
        adm_last = adm_last.with_columns([
            pl.col('proc_PCI').fill_null(0).cast(pl.Int8),
            pl.col('proc_CAG').fill_null(0).cast(pl.Int8),
            pl.col('prev_revasc').fill_null(0).cast(pl.Int8)
        ])
        adm_last = adm_last.sort(by=["subject_id", "hadm_id"]).unique(subset=["subject_id"], keep="last")

    if verbose:
        print(f'Number of attendances with prior PCI: {adm_last.filter(pl.col("proc_PCI") == 1).select("subject_id").n_unique()}, % of pts: {adm_last.filter(pl.col("proc_PCI") == 1).select("subject_id").n_unique() / adm_last.select("subject_id").n_unique() * 100:.2f}')
        print(f'Number of attendances with prior CAG: {adm_last.filter(pl.col("proc_CAG") == 1).select("subject_id").n_unique()}, % of pts: {adm_last.filter(pl.col("proc_CAG") == 1).select("subject_id").n_unique() / adm_last.select("subject_id").n_unique() * 100:.2f}')

    print("Collected prior procedures...")
    return adm_last.lazy() if use_lazy else adm_last

def read_outcomes_table(
    mimic4_path: str,
    adm_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the diagnoses table from MIMIC-IV for future acute cardiac diagnoses.

    Returns:
        pl.LazyFrame | pl.DataFrame: Outcomes table filtered and joined with admissions.
    """
    diag = pl.read_csv(
        os.path.join(mimic4_path, "diagnoses_icd.csv.gz"),
        columns=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        dtypes=[pl.Int64, pl.Int64, pl.Int16, pl.String, pl.Int16],
    )
    diag_mapping = read_d_icd_diagnoses_table(mimic4_path)
    if isinstance(adm_last, pl.LazyFrame):
        adm_last = adm_last.collect()

    diag = diag.join(diag_mapping, on="icd_code", how="inner")
    diag = diag.filter(pl.col("hadm_id").is_in(adm_last.select("hadm_id")))
    diag = diag.filter(pl.col("subject_id").is_in(adm_last.select("subject_id")))

    print("Collected diagnoses table for current episode..")
    return diag.lazy() if use_lazy else diag


def read_omr_table(
    mimic4_path: str,
    admits_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
    vitalsign_uom_map: dict = None,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the omr table (online medical records containing in-hospital measurements) from MIMIC-IV.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admits_last (pl.DataFrame | pl.LazyFrame): Final hospitalisations table for looking up historical data.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        vitalsign_uom_map (dict, optional): Mapping for measurement units.

    Returns:
        pl.LazyFrame | pl.DataFrame: OMR table in long format.
    """
    vitalsign_uom_map = {
        "Temperature": "°F",
        "Heart rate": "bpm",
        "Respiratory rate": "insp/min",
        "Oxygen saturation": "%",
        "Systolic blood pressure": "mmHg",
        "Diastolic blood pressure": "mmHg",
        "BMI": "kg/m²",
    }
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    omr = pl.read_csv(
        os.path.join(mimic4_path, "omr.csv.gz"),
        dtypes=[pl.Int64, pl.String, pl.Int64, pl.String, pl.String],
    )
    omr = omr.filter(pl.col("subject_id").is_in(admits_last.select("subject_id")))
    omr = omr.with_columns(
        pl.col("chartdate").str.strptime(pl.Date, "%Y-%m-%d").alias("charttime")
    )
    # Drop seq_num and chartdate
    omr = omr.drop(["seq_num", "chartdate"])

    ### Prepare hospital measures time-series
    omr = omr.join(
        admits_last.select(["subject_id", "edregtime"]),
        on="subject_id",
        how="left",
    )
    omr = omr.filter(((pl.col("charttime") <= pl.col("edregtime") + pl.duration(hours=3))))
    omr = omr.drop(["edregtime"])

    ### Requires reverting to pandas for string operations
    omr = omr.to_pandas()
    omr["result_name"] = np.where(
        omr["result_name"].str.contains("Blood Pressure"), "bp", omr["result_name"]
    )
    omr[["result_sysbp", "result_diabp"]] = omr["result_value"].str.split(
        "/", expand=True
    )
    omr["result_sysbp"] = pd.to_numeric(omr["result_sysbp"], errors="coerce")
    omr["result_diabp"] = pd.to_numeric(omr["result_diabp"], errors="coerce")
    omr["result_name"] = np.where(
        omr["result_name"].str.contains("BMI"), "bmi", omr["result_name"]
    )

    # Create separate rows for sysbp and diabp
    sysbp_measures = omr[["subject_id", "charttime", "result_sysbp"]].rename(
        columns={"result_sysbp": "value"}
    )
    sysbp_measures["label"] = "Systolic blood pressure"
    diabp_measures = omr[["subject_id", "charttime", "result_diabp"]].rename(
        columns={"result_diabp": "value"}
    )
    diabp_measures["label"] = "Diastolic blood pressure"

    # Concatenate the sysbp and diabp measures
    bp_measures = pd.concat([sysbp_measures, diabp_measures], axis=0)
    # Add BMI measurements
    bmi_measures = omr[omr["result_name"] == "bmi"][
        ["subject_id", "charttime", "result_value"]
    ].rename(columns={"result_value": "value"})
    bmi_measures["label"] = "BMI"
    omr = pd.concat([bp_measures, bmi_measures], axis=0)

    # Map the value_uom
    omr["valueuom"] = omr["label"].map(vitalsign_uom_map)
    omr["value"] = omr["value"].astype(np.float32)
    omr = pl.DataFrame(omr)

    return omr.lazy() if use_lazy else omr


def read_vitals_table(
    mimic4_ed_path: str,
    admits_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
    vitalsign_column_map: dict = None,
    vitalsign_uom_map: dict = None,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Read and preprocess the vitalsign table from MIMIC-IV-ED.

    Args:
        mimic4_ed_path (str): Path to directory containing MIMIC-IV ED module files.
        admits_last (pl.DataFrame | pl.LazyFrame): Final hospitalisations table for looking up historical data.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        vitalsign_column_map (dict, optional): Mapping for vital sign column names.
        vitalsign_uom_map (dict, optional): Mapping for vital sign units.

    Returns:
        pl.LazyFrame | pl.DataFrame: Vitals table in long format.
    """
    vitalsign_uom_map = {
        "Temperature": "°F",
        "Heart rate": "bpm",
        "Respiratory rate": "insp/min",
        "Oxygen saturation": "%",
        "Systolic blood pressure": "mmHg",
        "Diastolic blood pressure": "mmHg",
        "BMI": "kg/m²",
    }
    vitalsign_column_map = {
        "temperature": "Temperature",
        "heartrate": "Heart rate",
        "resprate": "Respiratory rate",
        "o2sat": "Oxygen saturation",
        "sbp": "Systolic blood pressure",
        "dbp": "Diastolic blood pressure",
    }

    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    vitals = pl.read_csv(
        os.path.join(mimic4_ed_path, "vitalsign.csv.gz"),
        dtypes=[pl.Int64, pl.Int64, pl.Datetime, pl.String, pl.String, pl.String],
        try_parse_dates=True,
    )

    ### Prepare ed vital signs measures in long table format
    vitals = vitals.filter(pl.col("subject_id").is_in(admits_last.select("subject_id")))
    vitals = vitals.drop(["ed_stay_id", "pain", "rhythm"])
    vitals = vitals.join(
        admits_last.select(["subject_id", "edregtime"]),
        on="subject_id",
        how="left",
    )
    vitals = vitals.filter(((pl.col("charttime") <= pl.col("edregtime") + pl.duration(hours=3))))
    vitals = vitals.drop(["edregtime"])
    vitals = vitals.rename(vitalsign_column_map)
    vitals = vitals.melt(
        id_vars=["subject_id", "charttime"],
        value_vars=[
            "Temperature",
            "Heart rate",
            "Respiratory rate",
            "Oxygen saturation",
            "Systolic blood pressure",
            "Diastolic blood pressure",
        ],
        variable_name="label",
        value_name="value",
    ).sort(by=["subject_id", "charttime"])
    vitals = vitals.drop_nulls("value").with_columns(pl.col("value").cast(pl.Float64))
    vitals = vitals.with_columns(
        pl.col("label").map_dict(vitalsign_uom_map).alias("valueuom")
    )

    return vitals.lazy() if use_lazy else vitals


def read_labevents_table(
    mimic4_path: str,
    admits_last: pl.DataFrame | pl.LazyFrame,
    include_items: None,
    items_path: str = "../config/lab_items.txt",
) -> pl.LazyFrame:
    """
    Read and preprocess the labevents table from MIMIC-IV.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admits_last (pl.DataFrame | pl.LazyFrame): Last admissions table for lookup.
        include_items (str): Path to file listing lab item IDs to include.

    Returns:
        pl.LazyFrame: Labevents table in long format.
    """
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    #  Load in csv using polars lazy API (requires table to be in csv format)
    labs_data = pl.scan_csv(
        os.path.join(mimic4_path, "labevents.csv"), try_parse_dates=True
    )
    d_items = (
        pl.read_csv(os.path.join(mimic4_path, "d_labitems.csv.gz"))
        .lazy()
        .select(["itemid", "label"])
    )
    # merge labitem id's with dict
    labs_data = labs_data.join(d_items, how="left", on="itemid")
    # select relevant columns
    labs_data = labs_data.select(
        ["subject_id", "hadm_id", "charttime", "itemid", "label", "value", "valueuom", "comments"]
    ).with_columns(
        charttime=pl.col("charttime").cast(pl.Datetime), linksto=pl.lit("labevents")
    )
    labs_data = labs_data.with_columns(pl.col("hadm_id").cast(pl.Int64))
    labs_data = labs_data.with_columns(pl.col("subject_id").cast(pl.Int64))

    # get eligible lab tests prior to current episode
    labs_data = labs_data.join(
        admits_last[["subject_id", "hadm_id", "edregtime"]]
        .lazy(),
        how="left",
        on=["subject_id", "hadm_id"],
    )
    labs_data = labs_data.filter((pl.col("charttime") <= pl.col("edregtime") + pl.duration(hours=3)))
    labs_data = labs_data.drop(["edregtime"])
    # get most common items (top 50 itemids by label)
    if include_items is None:
        lab_items = labs_data.groupby("itemid").agg(pl.count().alias("count")).sort("count", descending=True).head(50)
        ### Export items to file
        #lab_items.collect().write_csv("../config/lab_items.csv")
    if include_items is not None and items_path is not None:
        # read txt file containing list of ids
        with open(items_path) as f:
            lab_items = list(f.read().splitlines())

    labs_data = labs_data.filter(
        pl.col("itemid").cast(pl.Utf8).is_in(set(lab_items))
    )
    labs_data = labs_data.collect(streaming=True)
    labs_data = labs_data.sort(by=["subject_id", "hadm_id", "charttime"])
    labs_data = clean_labevents(labs_data)
    labs_data = labs_data.drop(["comments", "hadm_id"])

    return labs_data


def merge_events_table(
    vitals: pl.LazyFrame | pl.DataFrame,
    labs: pl.LazyFrame | pl.DataFrame,
    omr: pl.LazyFrame | pl.DataFrame,
    use_lazy: bool = False,
    verbose: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Merge vitals, labevents, and omr tables into a single events table for time-series modality.

    Args:
        vitals (pl.LazyFrame | pl.DataFrame): Vitals table.
        labs (pl.LazyFrame | pl.DataFrame): Labevents table.
        omr (pl.LazyFrame | pl.DataFrame): OMR table.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        verbose (bool): If True, print summary statistics.

    Returns:
        pl.LazyFrame | pl.DataFrame: Merged events table.
    """
    ### Collect dataframes if lazy
    if isinstance(vitals, pl.LazyFrame):
        vitals = vitals.collect()
    if isinstance(labs, pl.LazyFrame):
        labs = labs.collect(streaming=True)
    if isinstance(omr, pl.LazyFrame):
        omr = omr.collect()

    # Combine vitals into a single table
    vitals = vitals.with_columns(pl.lit("vitals_measurements").alias("linksto"))
    vitals = vitals.with_columns(pl.col("value").cast(pl.Float64).drop_nulls())
    vitals = vitals.sort(by=["subject_id", "charttime"]).unique(
        subset=["subject_id", "charttime", "label"], keep="last"
    )
    vitals = vitals.with_columns(pl.lit(None).alias("itemid"))
    ### Reorder itemid columns to be third
    # vitals = vitals.to_pandas()
    vitals = vitals.select(
        ["subject_id", "charttime", "itemid", "label", "value", "valueuom", "linksto"]
    )
    # vitals = pl.DataFrame(vitals)
    vitals = vitals.with_columns(pl.col("charttime").cast(pl.String))

    # Ensure labs also has charttime as String to match vitals
    if labs.schema["charttime"] != pl.Utf8:
        labs = labs.with_columns(pl.col("charttime").cast(pl.String))

    events = labs.vstack(vitals)
    if verbose:
        print(
            f"# collected vitals records: {vitals.shape[0]} across {vitals.select('subject_id').n_unique()} patients."
        )
        print(
            f"# collected lab records: {labs.shape[0]} across {labs.select('subject_id').n_unique()} patients."
        )
    return events.lazy() if use_lazy else events


def read_medications_table(
    mimic4_path: str,
    admits_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Get medication table from online administration record containing orders data.

    Args:
        mimic4_path (str): Path to directory containing MIMIC-IV module files.
        admits_last (pl.DataFrame | pl.LazyFrame): Last hospitalisations table for looking up historical data.
        use_lazy (bool): If True, return a Polars LazyFrame. Otherwise, return a DataFrame.
        top_n (int): Includes the top N most commonly prescribed medications as count features.

    Returns:
        pl.LazyFrame | pl.DataFrame: Admissions table with medication features.
    """
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    meds = pl.read_csv(
        os.path.join(mimic4_path, "emar.csv.gz"),
        dtypes=[pl.Int64, pl.Datetime, pl.String, pl.String],
        columns=["subject_id", "charttime", "medication", "event_txt"],
        try_parse_dates=True,
    )
    ### Link relevant medications and prepare for parsing
    meds = meds.filter(pl.col("subject_id").is_in(admits_last.select("subject_id")))
    meds = meds.join(
        admits_last.select(["subject_id", "edregtime"]), on="subject_id", how="left"
    )
    meds = meds.with_columns(pl.col("edregtime").cast(pl.Datetime))
    meds = meds.filter(pl.col("charttime") < pl.col("edregtime"))
    meds = meds.drop_nulls(subset=["medication", "event_txt", "charttime"])
    ### Filter correctly administered medications
    meds = meds.filter(
        pl.col("event_txt").is_in(["Administered", "Confirmed", "Started"])
    )
    ### Generate drug-level features and append to EHR data
    admits_last = prepare_medication_features(
        meds, admits_last, use_lazy=use_lazy
    )
    return admits_last.lazy() if use_lazy else admits_last


def save_multimodal_dataset(
    admits_last: pl.DataFrame | pl.LazyFrame,
    events: pl.DataFrame | pl.LazyFrame,
    use_events: bool = True,
    output_path: str = "../outputs/extracted_data",
):
    """
    Export datasets (EHR, events, notes) to CSV files for downstream processing.

    Args:
        admits_last (pl.DataFrame | pl.LazyFrame): Static EHR data.
        events (pl.DataFrame | pl.LazyFrame): Events time-series data.
        use_events (bool): If True, save events data.
        output_path (str): Directory to save the output files.

    Returns:
        None
    """
    #### Save EHR data (required)
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()
    admits_last.to_pandas().to_csv(
        os.path.join(output_path, "ehr_static.csv"), index=False
    )
    #### Save time-series and notes modality data
    if use_events:
        if isinstance(events, pl.LazyFrame):
            events = events.collect(streaming=True)
        events.write_csv(os.path.join(output_path, "events_ts.csv"))
