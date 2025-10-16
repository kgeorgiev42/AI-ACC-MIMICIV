import json
import os

import numpy as np
import pandas as pd
import polars as pl
import spacy
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils.functions import (
    contains_both_ltc_types,
    get_n_unique_values,
    get_train_split_summary,
    read_icd_mapping,
    rename_fields,
)

###############################
# EHR data preprocessing
###############################


def preproc_icd_module(
    diagnoses: pl.DataFrame | pl.LazyFrame,
    icd_map_path: str = "../config/icd9to10.txt",
    map_code_colname: str = "diagnosis_code",
    only_icd10: bool = True,
    cond_dict_path: str = "../outputs/icd10_codes.json",
    outcomes: bool = False,
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

    if isinstance(diagnoses, pl.LazyFrame):
        diagnoses = diagnoses.collect()

    cond_alias = "outcome_code" if outcomes else "ltc_code"

    def standardize_icd(mapping, df, root=False, icd_num=9):
        """Takes an ICD9 -> ICD10 mapping table and a module dataframe;
        adds column with converted ICD10 column"""

        def icd_9to10(icd):
            # If root is true, only map an ICD 9 -> 10 according to the ICD9's root (first 3 digits)
            if root:
                icd = icd[:3]
            try:
                # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
                return (
                    mapping.filter(pl.col(map_code_colname) == icd)
                    .select("icd10cm")
                    .to_series()[0]
                )
            except IndexError:
                # Handle case where no mapping is found for the ICD code
                return np.nan

        # Create new column with original codes as default
        col_name = "icd10_convert"
        if root:
            col_name = "root_" + col_name
        df = df.with_columns(pl.col("icd_code").alias(col_name).cast(pl.Utf8))

        # Convert ICD9 codes to ICD10 in a vectorized manner
        icd9_codes = (
            df.filter(pl.col("icd_version") == icd_num)
            .select("icd_code")
            .unique()
            .to_series()
            .to_list()
        )
        icd9_to_icd10_map = {code: icd_9to10(code) for code in icd9_codes}

        df = df.with_columns(
            pl.when(pl.col("icd_version") == icd_num)
            .then(
                pl.col("icd_code").apply(
                    lambda x: icd9_to_icd10_map.get(x, np.nan), return_dtype=pl.Utf8
                )
            )
            .otherwise(pl.col(col_name))
            .alias(col_name)
        )

        if only_icd10:
            # Column for just the roots of the converted ICD10 column
            df = df.with_columns(
                pl.col(col_name)
                .apply(
                    lambda x: x[:3] if isinstance(x, str) else np.nan,
                    return_dtype=pl.Utf8,
                )
                .alias("root")
            )

        return df

    # Optional ICD mapping if argument passed
    if icd_map_path:
        icd_map = read_icd_mapping(icd_map_path)
        diagnoses = standardize_icd(icd_map, diagnoses, root=True)
        diagnoses = diagnoses.filter(pl.col("root_icd10_convert").is_not_null())
        if verbose:
            print(
                "# unique ICD-10 codes (After converting ICD-9 to ICD-10)",
                diagnoses.select("root_icd10_convert").n_unique(),
            )
            print(
                "# unique ICD-10 codes (After clinical grouping ICD-10 codes)",
                diagnoses.select("root").n_unique(),
            )
            print("# Unique patients:  ", diagnoses.select("hadm_id").n_unique())

    diagnoses = diagnoses.select(
        ["subject_id", "hadm_id", "seq_num", "long_title", "root_icd10_convert"]
    )
    #### Create features for long-term chronic conditions
    if cond_dict_path:
        with open(cond_dict_path) as json_dict:
            cond_dict = json.load(json_dict)
        ### Initialise long-term condition column
        diagnoses = diagnoses.with_columns(
            pl.lit("Undefined").alias(cond_alias).cast(pl.Utf8)
        )
        print("Applying ICD-10 coding to diagnoses...")
        for ltc_group, codelist in tqdm(cond_dict.items()):
            # print("Group:", ltc_group, "Codes:", codelist)
            for code in codelist:
                diagnoses = diagnoses.with_columns(
                    pl.when(pl.col("root_icd10_convert").str.starts_with(code))
                    .then(pl.lit(ltc_group))
                    .otherwise(pl.col(cond_alias))
                    .alias(cond_alias)
                    .cast(pl.Utf8)
                )

    return diagnoses.lazy() if use_lazy else diagnoses


def get_diag_features(
    admits_last: pl.DataFrame | pl.LazyFrame,
    diagnoses: pl.DataFrame | pl.LazyFrame,
    cond_dict_path: str = "../outputs/icd10_codes.json",
    mm_cutoff: int = 1,
    cmm_cutoff: int = 3,
    use_mm: bool = True,
    outcomes: bool = False,
    verbose=True,
    use_lazy: bool = False,
) -> pl.DataFrame:
    """
    Generate features for long-term conditions and multimorbidity from ICD-10 diagnoses and custom LTC dictionary.

    Args:
        admits_last (pl.DataFrame | pl.LazyFrame): Admissions data.
        diagnoses (pl.DataFrame | pl.LazyFrame): ICD-10 Diagnoses data.
        cond_dict_path (str): Path to JSON with condition code groups.
        mm_cutoff (int): Threshold for multimorbidity.
        cmm_cutoff (int): Threshold for complex multimorbidity.
        verbose (bool): If True, print summary statistics.
        use_lazy (bool): If True, return a LazyFrame.

    Returns:
        pl.DataFrame or pl.LazyFrame: Admissions data with long-term condition count features.
    """

    if isinstance(diagnoses, pl.LazyFrame):
        diagnoses = diagnoses.collect()
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    code_alias = "outcome_code" if outcomes else "ltc_code"

    ### Comorbidity history
    diag_flat = diagnoses.filter(pl.col(code_alias) != "Undefined")
    if verbose:
        print(
            "Number of diagnoses recorded in hospital data:",
            diagnoses.shape[0],
            diagnoses["subject_id"].n_unique(),
        )

    ### Create list for each row in ltc_code column
    diag_flat = diag_flat.groupby("subject_id").agg(
        pl.col(code_alias).apply(set).alias(code_alias)
    )

    ### If dict is populated generate categorical columns for each long-term condition
    if cond_dict_path:
        with open(cond_dict_path) as json_dict:
            cond_dict = json.load(json_dict)
        for cond_code, _ in cond_dict.items():
            diag_flat = diag_flat.with_columns(
                pl.col(code_alias)
                .apply(lambda x, cond=cond_code: 1 if cond in x else 0)
                .alias(cond_code)
            )

    if use_mm:
        ### Create features for multimorbidity
        diag_flat = diag_flat.with_columns(
            [
                pl.col(code_alias)
                .apply(contains_both_ltc_types, return_dtype=pl.Int8)
                .alias("phys_men_multimorbidity"),
                pl.col(code_alias)
                .apply(len, return_dtype=pl.Int8)
                .alias("n_unique_conditions"),
                pl.when(pl.col(code_alias).apply(len, return_dtype=pl.Int8) > mm_cutoff)
                .then(1)
                .otherwise(0)
                .alias("is_multimorbid"),
                pl.when(pl.col(code_alias).apply(len, return_dtype=pl.Int8) > cmm_cutoff)
                .then(1)
                .otherwise(0)
                .alias("is_complex_multimorbid"),
            ]
        )

    ### Merge with base patient data
    admits_last = admits_last.join(diag_flat, on=["subject_id"], how="left")
    admits_last = admits_last.with_columns(
        [
            pl.col(col).cast(pl.Int8).fill_null(0)
            for col in diag_flat.drop(["subject_id", code_alias]).columns
        ]
    )
    admits_last = admits_last.sort(by=["subject_id"]).unique(subset=["subject_id"], keep='last')

    return admits_last.lazy() if use_lazy else admits_last


def transform_sensitive_attributes(ed_pts: pl.DataFrame) -> pl.DataFrame:
    """
    Map sensitive attributes (race, marital status) to predefined categories and types.

    Args:
        ed_pts (pl.DataFrame): Patient data.

    Returns:
        pl.DataFrame: Updated patient data.
    """

    ed_pts = ed_pts.with_columns(
        [
            pl.col("anchor_age").cast(pl.Int16),
            pl.when(
                pl.col("race")
                .str.to_lowercase()
                .str.contains("white|middle eastern|portuguese")
            )
            .then(pl.lit("White"))
            .when(
                pl.col("race").str.to_lowercase().str.contains("black|caribbean island")
            )
            .then(pl.lit("Black"))
            .when(
                pl.col("race")
                .str.to_lowercase()
                .str.contains("hispanic|south american")
            )
            .then(pl.lit("Hispanic/Latino"))
            .when(pl.col("race").str.to_lowercase().str.contains("asian"))
            .then(pl.lit("Asian"))
            .otherwise(pl.lit("Other"))
            .alias("race_group"),
            pl.col("marital_status").str.to_lowercase().str.to_titlecase(),
        ]
    )

    return ed_pts


def prepare_medication_features(
    medications: pl.DataFrame | pl.LazyFrame,
    admits_last: pl.DataFrame | pl.LazyFrame,
    use_lazy: bool = False,
) -> pl.DataFrame:
    """
    Generate count and temporal (days since prescription) features for drug-level medication history.

    Args:
        medications (pl.DataFrame | pl.LazyFrame): Medication data.
        admits_last (pl.DataFrame | pl.LazyFrame): Final hospitalisations data.
        use_lazy (bool): If True, return a LazyFrame.

    Returns:
        pl.DataFrame or pl.LazyFrame: Admissions data with medication count features.
    """
    if isinstance(medications, pl.LazyFrame):
        medications = medications.collect()
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    # Define medication patterns for vectorized search
    med_patterns = {
        "acei": ["benazepril", "lotensin", "captopril", "capoten", "cilazapril",
                 "inhibace", "enalapril", "vasotec", "renitec", "fosinopril", "monopril",
                 "imidapril", "tanatril", "lisinopril", "prinivil", "zestril",
                 "moexipril", "univasc", "perindopril", "coversyl", "aceon", "quinapril", "accupril",
                 "ramipril", "altace", "tritace", "trandolapril", "mavik", "odrik", "zofenopril", "zomen"],
        "arb": ["azilsartan", "candesartan", "eprosartan", "irbesartan",
                "losartan", "olmesartan", "telmisartan", "valsartan", "allisartan",
                "forasartan", "saprisartan", "sparsentan"],
        "bb": ["acebutolol", "sectral", "atenolol", "tenormin", "atecor", "atenomel", "atenamin", "atenix",
               "betaxolol", "bevantolol", "bisoprolol", "bucindolol", "carvedilol", "carteolol",
               "celiprolol", "esmolol", "labetalol", "metoprolol", "nadolol", "nebivolol",
               "oxprenolol", "penbutolol", "pindolol", "practolol", "propranolol", "sotalol",
               "timolol", "cardicor", "emcor", "monocor", "congescor", "coreg", "eucardic",
               "brevibloc", "trandate", "lopressor", "betaloc", "metocor", "corgard",
               "bedranol", "inderal", "beta-prograne", "beta_cardone", "tolerzide",
               "timolol", "nyogel", "prestim", "glau_opt", "combigan", "betimol"],
        "dapt": ["aspirin", "acetylsalicylic", "clopidogrel", "plavix", "effient", "prasugrel",
                 "ticagrelor", "brilinta", "dipyridamole", "cangrelor", "cilostazol"],
        "aspirin": ["aspirin", "acetylsalicylic", "paracetamol", "ecotrin",
                    "ascriptin", "easprin", "bayer", "halfprin", "healthprin", "measurin",
                    "bufferin", "empirin"]
    }

    # Use Polars for preprocessing
    # Convert charttime to datetime if it's a string
    if medications["charttime"].dtype == pl.Utf8:
        medications = medications.with_columns(pl.col("charttime").str.to_datetime())

    # Convert edregtime to datetime if it's a string
    if medications["edregtime"].dtype == pl.Utf8:
        medications = medications.with_columns(pl.col("edregtime").str.to_datetime())

    # Clean medication names
    medications = medications.with_columns(
        pl.col("medication").str.to_lowercase().str.strip_chars().str.replace_all(" ", "_").str.replace_all("-", "_")
    )

    # Filter medications before edregtime
    medications = medications.filter(pl.col("charttime") < pl.col("edregtime"))

    # Create medication class indicators using vectorized operations
    med_class_exprs = []
    for med_class, med_list in med_patterns.items():
        pattern = "|".join(med_list)
        med_class_exprs.append(
            pl.col("medication").str.contains(pattern).cast(pl.Int8).alias(f"is_{med_class}")
        )

    medications = medications.with_columns(med_class_exprs)

    # Aggregate prescriptions by subject_id and medication class in a single operation
    # Group by subject_id and sum all medication class indicators
    aggregation_exprs = [pl.col(f"is_{med_class}").sum().alias(f"n_presc_{med_class}")
                         for med_class in med_patterns.keys()]

    meds_summary = medications.group_by("subject_id").agg(aggregation_exprs)

    # Join with admits_last using Polars (faster than pandas)
    admits_last = admits_last.join(meds_summary, on=["subject_id"], how="left")
    #admits_last = admits_last.sort(by=["subject_id"]).unique(subset=["subject_id"], keep='last')

    # Fill nulls and cast to Int32 to handle larger prescription counts
    presc_cols = [f"n_presc_{med_class}" for med_class in med_patterns.keys()]
    admits_last = admits_last.with_columns([
        pl.col(col).fill_null(0).cast(pl.Int32) for col in presc_cols
    ])

    return admits_last.lazy() if use_lazy else admits_last


def encode_categorical_features(ehr_data: pl.DataFrame) -> pl.DataFrame:
    """
    Apply one-hot encoding to categorical features in EHR data.

    Args:
        ehr_data (pl.DataFrame): Static EHR dataset.

    Returns:
        pl.DataFrame: Transformed EHR data.
    """

    # prepare attribute features for one-hot-encoding
    ehr_data = ehr_data.with_columns(
        [
            pl.when(pl.col("race_group") == "Hispanic/Latino")
            .then(pl.lit("Hispanic_Latino"))
            .otherwise(pl.col("race_group"))
            .alias("race_group"),
            pl.when(pl.col("gender") == "F")
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("gender_F"),
        ]
    )
    ehr_data = ehr_data.to_dummies(
        columns=["race_group", "marital_status", "insurance"]
    )
    ehr_data = ehr_data.drop(["race", "gender"])
    ### Drop temporal columns if only a few are retained (for MLP classifier stability)
    ehr_data = ehr_data.drop(
        [col for col in ehr_data.columns if col.startswith("dsf_")]
    )
    ehr_data = ehr_data.drop(
        [col for col in ehr_data.columns if col.startswith("dsl_")]
    )
    return ehr_data


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
    ehr_lookup = ehr_data.select(["subject_id"] + lookup_list)
    ehr_data = ehr_data.drop(lookup_list)
    print(f"Saving lookup fields in EHR data to {lookup_output_path}")
    ## Create folder if it does not exist
    if not os.path.exists(lookup_output_path):
        os.makedirs(lookup_output_path)
    ehr_lookup.write_csv(os.path.join(lookup_output_path, "ehr_lookup.csv"))
    return ehr_data


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
    ### Specify features to save
    ehr_save = ehr_data.select(["subject_id"] + feats_to_save)
    ehr_data = ehr_data.drop(["subject_id"] + feats_to_save)
    ### Generate a linear correlation matrix
    corr_matrix = ehr_data.to_pandas().corr(method=method)
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    for i in tqdm(iters, desc="Dropping highly correlated features..."):
        for j in range(i + 1):
            item = corr_matrix.iloc[j : (j + 1), (i + 1) : (i + 2)]
            colname = item.columns
            val = abs(item.values)
            if val >= threshold:
                drop_cols.append(colname.values[0])

    to_drop = list(set(drop_cols))
    ehr_data = ehr_data.drop(to_drop)
    ehr_data = ehr_save.select(["subject_id"]).hstack(ehr_data)
    ehr_data = ehr_data.join(ehr_save, on="subject_id", how="left")

    if verbose:
        print(f"Dropped {len(to_drop)} highly correlated features.")
        print("-------------------------------------")
        print("Full list of dropped features:", to_drop)
        print("-------------------------------------")
        print(
            f"Final number of EHR features: {ehr_data.shape[1]}/{len(to_drop)+ehr_data.shape[1]}"
        )

    return ehr_data

def undersample_set(x_data, y_data, set_name, outcome_col, seed: int=42, verbose: bool = True):
    """Apply undersampling to a given dataset."""
    # Combine features and labels for undersampling
    combined = pd.concat([x_data, y_data], axis=1)

    # Separate positive and negative classes
    positive_class = combined[combined[outcome_col] == 1]
    negative_class = combined[combined[outcome_col] == 0]

    n_positive = len(positive_class)
    n_negative = len(negative_class)

    if verbose:
        print(f"{set_name} - Before undersampling - Positive: {n_positive}, Negative: {n_negative}")

    # Undersample negative class to match positive class size
    if n_negative > n_positive:
        negative_class_sampled = negative_class.sample(
            n=n_positive, random_state=seed, replace=False
        )
        combined_sampled = pd.concat([positive_class, negative_class_sampled], axis=0)
        combined_sampled = combined_sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Split back into features and labels
        x_sampled = combined_sampled.drop(columns=[outcome_col, "gender", "race_group"])
        y_sampled = combined_sampled[[outcome_col, "gender", "race_group"]]

        if verbose:
            print(f"{set_name} - After undersampling - Positive: {len(combined_sampled[combined_sampled[outcome_col] == 1])}, Negative: {len(combined_sampled[combined_sampled[outcome_col] == 0])}")

        return x_sampled, y_sampled
    else:
        if verbose:
            print(f"{set_name} - No undersampling needed - positive class is equal or larger than negative class")
        return x_data, y_data

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
    undersample_train: bool = False,
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
        undersample_train (bool): If True, randomly undersample the negative class in training and validation sets to match positive class.
        verbose (bool): If True, print summary.

    Returns:
        dict: Dictionary with train, val, and test DataFrames.
    """
    ### Define display dictionary
    disp_dict = {
        "anchor_age": "Age",
        "gender": "Gender",
        "race_group": "Ethnicity",
        "insurance": "Insurance",
        "marital_status": "Marital status",
        "in_hosp_death": "In-hospital death",
        "ext_stay_7": "Extended stay",
        "non_home_discharge": "Non-home discharge",
        "icu_admission": "ICU admission",
        "is_multimorbid": "Multimorbidity",
        "is_complex_multimorbid": "Complex multimorbidity",
    }
    cont_cols = ["Age"]
    ### List non-normally distributed columns here for re-scaling
    ## TODO: Would need to move this to an appropriate config file
    nn_cols = [
        "total_n_presc",
        "n_unique_conditions",
        "n_presc_acetaminophen",
        "n_presc_acyclovir",
        "n_presc_albuterol_neb_soln",
        "n_presc_amlodipine",
        "n_presc_apixaban",
        "n_presc_aspirin",
        "n_presc_atorvastatin",
        "n_presc_calcium_carbonate",
        "n_presc_carvedilol",
        "n_presc_cefepime",
        "n_presc_ceftriaxone",
        "n_presc_docusate_sodium",
        "n_presc_famotidine",
        "n_presc_folic_acid",
        "n_presc_furosemide",
        "n_presc_gabapentin",
        "n_presc_heparin",
        "n_presc_hydralazine",
        "n_presc_hydromorphone_dilaudid",
        "n_presc_insulin",
        "n_presc_ipratropium_albuterol_neb",
        "n_presc_lactulose",
        "n_presc_levetiracetam",
        "n_presc_levothyroxine_sodium",
        "n_presc_lisinopril",
        "n_presc_lorazepam",
        "n_presc_metoprolol_succinate_xl",
        "n_presc_metoprolol_tartrate",
        "n_presc_metronidazole",
        "n_presc_midodrine",
        "n_presc_morphine_sulfate",
        "n_presc_multivitamins",
        "n_presc_omeprazole",
        "n_presc_ondansetron",
        "n_presc_oxycodone",
        "n_presc_pantoprazole",
        "n_presc_piperacillin_tazobactam",
        "n_presc_polyethylene_glycol",
        "n_presc_potassium_chloride",
        "n_presc_prednisone",
        "n_presc_rifaximin",
        "n_presc_senna",
        "n_presc_sevelamer_carbonate",
        "n_presc_tacrolimus",
        "n_presc_thiamine",
        "n_presc_vancomycin",
        "n_presc_vitamin_d",
        "n_presc_warfarin",
        "pon_nutrition",
        "pon_cardiology",
        "pon_respiratory",
        "pon_neurology",
        "pon_radiology",
        "pon_tpn",
        "pon_hemodialysis",
    ]

    cat_cols = [
        "In-hospital death",
        "Extended stay",
        "Non-home discharge",
        "ICU admission",
        "Multimorbidity",
        "Complex multimorbidity",
    ]

    ### Set stratification columns to include sensitive attributes + target outcome
    ehr_data = ehr_data.to_pandas()
    if stratify:
        strat_target = pd.concat(
            [ehr_data[outcome_col], ehr_data["gender"], ehr_data["race_group"]], axis=1
        )
        split_target = ehr_data.drop([outcome_col, "gender", "race_group"], axis=1)
        ### Generate split dataframes
        train_x, test_x, train_y, test_y = train_test_split(
            split_target,
            strat_target,
            test_size=(1 - train_ratio),
            random_state=seed,
            stratify=strat_target,
        )
        val_x, test_x, val_y, test_y = train_test_split(
            test_x,
            test_y,
            test_size=test_ratio / (test_ratio + val_ratio),
            random_state=seed,
            stratify=test_y,
        )
    else:
        train_x, test_x, train_y, test_y = train_test_split(
            ehr_data.drop([outcome_col], axis=1),
            ehr_data[outcome_col],
            test_size=(1 - train_ratio),
            random_state=seed,
        )
        val_x, test_x, val_y, test_y = train_test_split(
            test_x,
            test_y,
            test_size=test_ratio / (test_ratio + val_ratio),
            random_state=seed,
        )

    ### Re-scale EHR data for MLP classifier
    scaler = MinMaxScaler()

    # Apply random undersampling to training and validation sets
    if undersample_train:
        # Apply undersampling to training set
        train_x, train_y = undersample_set(train_x, train_y, "Training set", outcome_col=outcome_col, seed=seed, verbose=verbose)

        # Apply undersampling to validation set
        val_x, val_y = undersample_set(val_x, val_y, "Validation set", outcome_col=outcome_col, seed=seed, verbose=verbose)

    train_x[nn_cols] = scaler.fit_transform(train_x[nn_cols])
    val_x[nn_cols] = scaler.transform(val_x[nn_cols])  # Apply transformation to val_x
    test_x[nn_cols] = scaler.transform(
        test_x[nn_cols]
    )  # Apply transformation to test_x
    train_x = pd.concat([train_x, train_y], axis=1)
    val_x = pd.concat([val_x, val_y], axis=1)
    test_x = pd.concat([test_x, test_y], axis=1)
    train_x["set"] = "train"
    val_x["set"] = "val"
    test_x["set"] = "test"
    ### Print summary statistics
    if verbose:
        print(
            f"Created split with {train_x.shape[0]}({round(train_x.shape[0]/len(ehr_data), 2)*100}%) samples in train, {val_x.shape[0]}({round(val_x.shape[0]/len(ehr_data), 2)*100}%) samples in validation, and {test_x.shape[0]}({round(test_x.shape[0]/len(ehr_data), 2)*100}%) samples in test."
        )
        print("Getting summary statistics for split...")
        get_train_split_summary(
            train_x,
            val_x,
            test_x,
            outcome_col,
            output_summary_path,
            cont_cols,
            nn_cols,
            disp_dict,
            cat_cols,
            verbose=verbose,
        )
        print(f"Saving train/val/test split IDs to {output_path}")

    ### Save patient IDs
    train_x[["subject_id"]].to_csv(
        os.path.join(output_path, "training_ids_" + outcome_col + ".csv"), index=False
    )
    val_x[["subject_id"]].to_csv(
        os.path.join(output_path, "validation_ids_" + outcome_col + ".csv"), index=False
    )
    test_x[["subject_id"]].to_csv(
        os.path.join(output_path, "testing_ids_" + outcome_col + ".csv"), index=False
    )

    return {"train": train_x, "val": val_x, "test": test_x}


###############################
# Notes preprocessing
###############################


def clean_notes(notes: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Clean notes data by removing special characters and extra whitespaces.

    Args:
        notes (pl.DataFrame | pl.LazyFrame): Notes data.

    Returns:
        pl.DataFrame or pl.LazyFrame: Cleaned notes data.
    """
    # Remove __ and any extra whitespaces
    notes = notes.with_columns(
        target=pl.col("target")
        .str.replace_all(r"___", " ")
        .str.replace_all(r"\s+", " ")
    )
    # notes = notes.with_columns(target=pl.col("target").str.replace_all(r"\s+", " "))
    return notes


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
    embeddings_dict = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    nlp = spacy.load("en_core_sci_md", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")
    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_Discharge_Summary_BERT"
    )
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT").to(
        device
    )
    # randomly downsample notes for testing
    # notes = notes.sample(fraction=0.05)
    for row in tqdm(
        notes.iter_rows(named=True),
        desc="Generating notes embeddings with ClinicalBERT...",
        total=notes.height,
    ):
        subj_id = row["subject_id"]
        text = row["target"]

        # Turn text into sentences
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        # Tokenize all sentences at once
        inputs = tokenizer(
            sentences,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_attention_mask=True,
        ).to(device)

        # Generate embeddings for all sentences in a single forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            sentence_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        if sentence_embeddings.size > 0:
            embeddings = np.mean(sentence_embeddings, axis=0)
        else:
            embeddings = np.zeros((768,))  # Handle case with no sentences

        # Map each sentence to its embedding
        sentence_embedding_pairs = list(zip(sentences, embeddings, strict=False))

        # Store the mapping in the dictionary
        embeddings_dict[subj_id] = sentence_embedding_pairs

    return embeddings_dict


###############################
# Time-series preprocessing
###############################


def clean_labevents(labs_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Clean lab events by removing non-integer values and outliers.

    Args:
        labs_data (pl.LazyFrame): Lab events data.

    Returns:
        pl.LazyFrame: Cleaned lab events.
    """
    labs_data = labs_data.with_columns(
        pl.col("label")
        .str.to_lowercase()
        .str.replace(" ", "_")
        .str.replace(",", "")
        .str.replace('"', "")
        .str.replace(" ", "_"),
        pl.col("charttime").cast(pl.Utf8).str.replace("T", " ").str.strip_chars(),
    )
    lab_events = labs_data.with_columns(
        value=pl.when(pl.col("value") == ".").then(None).otherwise(pl.col("value"))
    )
    lab_events = lab_events.with_columns(
        value=pl.when(pl.col("value").str.contains("_|<|ERROR"))
        .then(None)
        .otherwise(pl.col("value"))
        .cast(
            pl.Float64, strict=False
        )  # Attempt to cast to Float64, set invalid values to None
    )
    labs_data = labs_data.drop_nulls()

    # Remove outliers using 5 std from mean
    '''
    lab_events = lab_events.with_columns(
        mean=pl.col("value").mean().over(pl.count("label"))
    )
    lab_events = lab_events.with_columns(
        std=pl.col("value").std().over(pl.count("label"))
    )

    lab_events = lab_events.filter(
        (pl.col("value") <= pl.col("mean") + pl.col("std") * 100)
        & (pl.col("value") >= pl.col("mean") - pl.col("std") * 100)
    ).drop(["mean", "std"])
    '''

    lab_troponin = lab_events.filter(pl.col("label").str.contains("troponin"))
    print(f"Total troponin measurements: {lab_troponin.height}")

    #lab_events = lab_events.collect(streaming=True)

    # Cardiovascular-specific lab value cleaning
    lab_events = clean_specific_lab_values(lab_events)

    # Extract troponin T measures
    lab_events = extract_troponin_t_measures(lab_events)

    # Replace overlapping lab labels
    lab_events = lab_events.with_columns(
        pl.when(pl.col("label").is_in(['white_blood_cells', 'wbc', 'wbc_count']))
        .then(pl.lit('wbc'))
        .otherwise(pl.col("label"))
        .alias("label")
    )
    lab_events = lab_events.with_columns(
        pl.when(pl.col("label").is_in(['estimated_gfr_(mdrd equation)', 'egfr', 'gfr', 'egfr_(ckd-epi)']))
        .then(pl.lit('eGFR'))
        .otherwise(pl.col("label"))
        .alias("label")
    )
    lab_events = lab_events.with_columns(
        pl.when(pl.col("label").is_in(['creatine_kinase_(ck)', 'ck', 'creatine_kinase']))
        .then(pl.lit('creatine_kinase'))
        .otherwise(pl.col("label"))
        .alias("label")
    )
    lab_events = lab_events.with_columns(
        pl.when(pl.col("label").is_in(['creatine_kinase_mb isoenzyme', 'ck-mb', 'ck_mb', 'creatine_kinase_mb']))
        .then(pl.lit('creatine_kinase_mb'))
        .otherwise(pl.col("label"))
        .alias("label")
    )

    return lab_events


def clean_specific_lab_values(labs_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Apply specific cleaning rules for lab values.

    Args:
        labs_data (pl.LazyFrame): Lab events data with 'label', 'value', and 'comments' columns.

    Returns:
        pl.LazyFrame: Lab data with cleaned creatine kinase, hemoglobin, and eGFR values.
    """

    ## Troponin T
    cleaned_data = labs_data.with_columns(
        pl.when(pl.col("label") == "troponin_t")
        .then(
            pl.when(
                pl.col("value").is_nan() &
                (pl.col("comments").str.starts_with('<') | pl.col("comments").str.contains('(?i)LESS'))
            )
            .then(0.01)
            .when(
                pl.col("value").is_nan() &
                (pl.col("comments").str.starts_with('>') | pl.col("comments").str.contains('(?i)GREATER'))
            )
            .then(25)
            .otherwise(pl.col("value"))
        )
        .otherwise(pl.col("value"))
        .alias("value")
    )

    # Hemoglobin specific cleaning
    cleaned_data = cleaned_data.with_columns(
        pl.when(pl.col("label").str.contains("hemoglobin"))
        .then(
            pl.when(pl.col("value").is_nan() & pl.col("comments").str.contains("(?i)<|less"))
            .then(3.0)  # Lower detection limit
            .when(pl.col("value").is_nan() & pl.col("comments").str.contains("(?i)>|greater"))
            .then(25.0)  # Upper detection limit
            .when(pl.col("value") < 2.0)
            .then(None)  # Below physiological minimum
            .when(pl.col("value") > 25.0)
            .then(None)  # Above physiological maximum
            .when((pl.col("value") >= 100) & (pl.col("value") <= 250))
            .then(pl.col("value") / 10)  # Convert g/L to g/dL
            .otherwise(pl.col("value"))
        )
        .otherwise(pl.col("value"))
        .alias("value")
    )

    # eGFR specific cleaning
    cleaned_data = cleaned_data.with_columns(
        pl.when(pl.col("label").str.contains("gfr"))
        .then(
            pl.when(pl.col("value").is_nan() & pl.col("comments").str.contains("(?i)<|less"))
            .then(5.0)  # Lower detection limit
            .when(pl.col("value").is_nan() & pl.col("comments").str.contains("(?i)>|greater"))
            .then(150.0)  # Upper detection limit
            .when(pl.col("comments").str.contains("(?i)>60|greater.*60"))
            .then(90.0)  # Use midpoint for >60 reporting
            .when(pl.col("value") < 0)
            .then(None)  # Remove negative values
            .when(pl.col("value") > 200)
            .then(None)  # Remove extreme outliers (normal: 90-120 mL/min/1.73m²)
            .otherwise(pl.col("value"))
        )
        .otherwise(pl.col("value"))
        .alias("value")
    )

    return cleaned_data


def extract_troponin_t_measures(labs_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Extract first, second, and third troponin T measures per hospital admission (hadm_id)
    ordered by charttime, and calculate delta troponin T (difference between highest and lowest).

    Creates new rows in the labs_data with specific labels:
    - 'first_troponin_t': First troponin T measurement
    - 'second_troponin_t': Second troponin T measurement
    - 'third_troponin_t': Third troponin T measurement
    - 'troponin_t_delta': Difference between max and min values

    Args:
        labs_data (pl.LazyFrame): Lab events data with columns: hadm_id, charttime, label, value

    Returns:
        pl.LazyFrame: Original labs_data with additional troponin T feature rows appended
    """
    # Filter for troponin T measurements and ensure required columns exist
    troponin_data = labs_data.filter(
        pl.col("label").str.contains("troponin_t")
    ).filter(
        pl.col("value").is_not_null() &
        pl.col("hadm_id").is_not_null() &
        pl.col("charttime").is_not_null()
    )

    # Convert charttime to datetime if it's not already
    troponin_data = troponin_data.with_columns(
        pl.col("charttime").str.to_datetime(strict=False).alias("charttime")
    )

    # Sort by hadm_id and charttime to get chronological order
    troponin_sorted = troponin_data.sort(["hadm_id", "charttime"])

    # Add row number within each admission to identify 1st, 2nd, 3rd measurements
    troponin_numbered = troponin_sorted.with_columns(
        pl.int_range(pl.len()).over("hadm_id").alias("measurement_order")
    )

    # Create pivot-like structure for first, second, third measurements
    troponin_pivot = troponin_numbered.group_by("hadm_id").agg([
        pl.col("value").filter(pl.col("measurement_order") == 0).first().alias("first_troponin_t"),
        pl.col("value").filter(pl.col("measurement_order") == 1).first().alias("second_troponin_t"),
        pl.col("value").filter(pl.col("measurement_order") == 2).first().alias("third_troponin_t"),
        pl.col("value").min().alias("troponin_t_min"),
        pl.col("value").max().alias("troponin_t_max"),
        pl.col("value").len().alias("troponin_t_count"),
        pl.col("subject_id").first().alias("subject_id"),
        pl.col("charttime").first().alias("base_charttime")  # Use first charttime as reference
    ])

    # Calculate delta troponin T (difference between highest and lowest)
    troponin_features = troponin_pivot.with_columns(
        pl.when(pl.col("troponin_t_count") > 1)
        .then(pl.col("troponin_t_max") - pl.col("troponin_t_min"))
        .otherwise(None)
        .alias("troponin_t_delta")
    )

    # Create new rows for each troponin feature

    # Create base template with required columns filled from troponin_features
    base_template = troponin_features.select([
        "hadm_id", "subject_id", "base_charttime"
    ]).with_columns([
        pl.col("base_charttime").alias("charttime"),
        pl.lit(None).alias("comments")  # Add comments column if it exists in original
    ])

    # First troponin T
    first_troponin_rows = base_template.join(
        troponin_features.select(["hadm_id", "first_troponin_t"]),
        on="hadm_id", how="inner"
    ).filter(
        pl.col("first_troponin_t").is_not_null()
    ).with_columns([
        pl.lit("first_troponin_t").alias("label"),
        pl.col("first_troponin_t").alias("value")
    ]).drop("first_troponin_t")

    # Second troponin T
    second_troponin_rows = base_template.join(
        troponin_features.select(["hadm_id", "second_troponin_t"]),
        on="hadm_id", how="inner"
    ).filter(
        pl.col("second_troponin_t").is_not_null()
    ).with_columns([
        pl.lit("second_troponin_t").alias("label"),
        pl.col("second_troponin_t").alias("value")
    ]).drop("second_troponin_t")

    # Third troponin T
    third_troponin_rows = base_template.join(
        troponin_features.select(["hadm_id", "third_troponin_t"]),
        on="hadm_id", how="inner"
    ).filter(
        pl.col("third_troponin_t").is_not_null()
    ).with_columns([
        pl.lit("third_troponin_t").alias("label"),
        pl.col("third_troponin_t").alias("value")
    ]).drop("third_troponin_t")

    # Troponin T delta
    delta_troponin_rows = base_template.join(
        troponin_features.select(["hadm_id", "troponin_t_delta"]),
        on="hadm_id", how="inner"
    ).filter(
        pl.col("troponin_t_delta").is_not_null()
    ).with_columns([
        pl.lit("troponin_t_delta").alias("label"),
        pl.col("troponin_t_delta").alias("value")
    ]).drop("troponin_t_delta")

    # Combine all feature rows
    all_troponin_features = pl.concat([
        first_troponin_rows,
        second_troponin_rows,
        third_troponin_rows,
        delta_troponin_rows
    ], how="vertical")

    # Add any missing columns that exist in original labs_data but not in our feature rows
    original_cols = set(labs_data.columns)
    feature_cols = set(all_troponin_features.columns)
    missing_cols = original_cols - feature_cols

    if missing_cols:
        for col in missing_cols:
            all_troponin_features = all_troponin_features.with_columns(
                pl.lit(None).alias(col)
            )

    # Ensure column order matches original
    all_troponin_features = all_troponin_features.select(labs_data.columns)

    # Convert charttime back to String to match original labs_data format
    all_troponin_features = all_troponin_features.with_columns(
        pl.col("charttime").cast(pl.Utf8).alias("charttime")
    )

    # Append new troponin feature rows to original labs_data
    enhanced_labs_data = pl.concat([labs_data, all_troponin_features], how="vertical")

    return enhanced_labs_data


def clean_vitals(vitals_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Clean extracted vital signs measurements, clipping them to plausible ranges and converting Celsius to Fahrenheit where appropriate.

    Args:
        vitals_data (pl.LazyFrame): Vital signs data.

    Returns:
        pl.LazyFrame: Cleaned vital signs data.
    """

    # Range definitions
    celc_conversion = 9 / 5 + 32
    celc_range = (25, 45)  # Plausible Celsius range for temperature
    fahr_range = (85, 110)  # Plausible Fahrenheit range for temperature
    sys_bp_range = (70, 220)  # Plausible range for systolic blood pressure
    dias_bp_range = (40, 130)  # Plausible range for diastolic blood pressure
    heart_rate_range = (30, 220)  # Plausible range for heart rate
    oxygen_saturation_range = (70, 100)  # Plausible range for oxygen saturation
    respiratory_rate_range = (5, 60)  # Plausible range for respiratory rate

    # Convert 'Temperature' values in Celsius to Fahrenheit (if in plausible Celsius range)
    vitals_cleaned = vitals_data.with_columns(
        pl.when((pl.col("label") == "Temperature") & (pl.col("value") < celc_range[1]) & (pl.col("value") > celc_range[0]))
        .then(pl.col("value") * celc_conversion)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    # Clip Systolic blood pressure values to [70, 220]
    vitals_cleaned = vitals_cleaned.with_columns(
        pl.when(pl.col("label") == "Systolic blood pressure")
        .then(pl.col("value").clip(sys_bp_range[0], sys_bp_range[1]))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    # Clip Diastolic blood pressure values to [40, 130]
    vitals_cleaned = vitals_cleaned.with_columns(
        pl.when(pl.col("label") == "Diastolic blood pressure")
        .then(pl.col("value").clip(dias_bp_range[0], dias_bp_range[1]))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    # Clip Heart rate values to [30, 220]
    vitals_cleaned = vitals_cleaned.with_columns(
        pl.when(pl.col("label") == "Heart rate")
        .then(pl.col("value").clip(heart_rate_range[0], heart_rate_range[1]))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    # Clip Oxygen saturation values to [70, 100]
    vitals_cleaned = vitals_cleaned.with_columns(
        pl.when(pl.col("label") == "Oxygen saturation")
        .then(pl.col("value").clip(oxygen_saturation_range[0], oxygen_saturation_range[1]))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    # Clip Respiratory rate values to [5, 60]
    vitals_cleaned = vitals_cleaned.with_columns(
        pl.when(pl.col("label") == "Respiratory rate")
        .then(pl.col("value").clip(respiratory_rate_range[0], respiratory_rate_range[1]))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    # Drop invalid measurements for temperature only
    vitals_cleaned = vitals_cleaned.filter(
        (pl.col("label") != "Temperature") |
        ((pl.col("value") >= fahr_range[0]) & (pl.col("value") <= fahr_range[1]))
    )
    return vitals_cleaned

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
    events = events.with_columns(
        elapsed=((pl.col("charttime") - starttime) / pl.duration(hours=1)).round(1)
    )

    # reorder columns
    if remove_charttime:
        events = events.drop("charttime")

    return events


def convert_events_to_timeseries(events: pl.DataFrame) -> pl.DataFrame:
    """
    Convert long-form events to wide-form time-series.

    Args:
        events (pl.DataFrame): Long-form events.

    Returns:
        pl.DataFrame: Wide-form time-series.
    """

    metadata = (
        events.select(["charttime", "label", "value", "linksto"])
        .sort(by=["charttime", "label", "value"])
        .unique(subset=["charttime"], keep="last")
        .sort(by="charttime")
    )

    # get unique label, values and charttimes
    timeseries = (
        events.select(["charttime", "label", "value"])
        .sort(by=["charttime", "label", "value"])
        .unique(subset=["charttime", "label"], keep="last")
    )

    # pivot into wide-form format
    timeseries = timeseries.pivot(
        index="charttime", columns="label", values="value"
    ).sort(by="charttime")

    # join any metadata remaining
    timeseries = timeseries.join(
        metadata.select(["charttime", "linksto"]), on="charttime", how="inner"
    )
    return timeseries


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

    data_dict = {}
    col_dict = {}
    n = 0
    filter_by_nb_events = 0
    missing_event_src = 0
    filter_by_elapsed_time = 0
    n_src = ts_data.n_unique("linksto")
    ehr_lkup = ehr_static.drop("subject_id")
    ehr_lkup = ehr_lkup.drop(outcomes)
    col_dict["static_cols"] = ehr_lkup.columns
    col_dict["dynamic0_cols"] = vitals_lkup
    col_dict["notes_cols"] = ["sentence", "embedding"]

    feature_map, freq = _prepare_feature_map_and_freq(ts_data, vitals_freq, lab_freq)
    min_events = 1 if min_events is None else int(min_events)
    max_events = 1e6 if max_events is None else int(max_events)

    ## Standardize vital signs data between 0 and 1
    if standardize:
        ts_data = _standardize_data(ts_data)
    ts_data = ts_data.sort(by=["subject_id", "charttime"])
    ehr_regtime = ehr_regtime.sort(by=["subject_id", "edregtime"])

    for id_val in tqdm(
        ts_data.unique("subject_id").get_column("subject_id").to_list(),
        desc="Generating patient-level data...",
    ):
        pt_events = ts_data.filter(pl.col("subject_id") == id_val)
        edregtime = (
            ehr_regtime.filter(pl.col("subject_id") == id_val)
            .select("edregtime")
            .head(1)
            .item()
        )
        ehr_sel = ehr_static.filter(pl.col("subject_id") == id_val)

        if pt_events.n_unique("linksto") < n_src:
            missing_event_src += 1
            continue

        write_data, ts_data_list, s_ec, s_et = _process_patient_events(
            pt_events,
            feature_map,
            freq,
            ehr_static,
            edregtime,
            min_events,
            max_events,
            impute,
            include_dyn_mean,
            no_resample,
            max_elapsed,
        )

        if s_ec:
            filter_by_nb_events += 1
            continue

        if s_et:
            filter_by_elapsed_time += 1
            continue

        if write_data:
            ## Encode count features for training
            ehr_cur = ehr_sel.drop(outcomes)
            ehr_cur = ehr_cur.drop("subject_id").to_numpy()
            data_dict[id_val] = {"static": ehr_cur}
            for outcome in outcomes:
                data_dict[id_val][outcome] = (
                    ehr_sel.select(outcome).cast(pl.Int8).to_numpy()
                )
            for _, ts in enumerate(ts_data_list):
                key = "dynamic_0" if ts.columns == vitals_lkup else "dynamic_1"
                if key == "dynamic_1" and "dynamic1_cols" not in col_dict.keys():
                    col_dict["dynamic1_cols"] = ts.columns
                data_dict[id_val][key] = ts.to_numpy()
            n += 1

    if verbose:
        _print_summary(
            n, filter_by_nb_events, missing_event_src, filter_by_elapsed_time
        )

    return data_dict, col_dict


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
    feature_map: dict = {}
    freq: dict = {}
    for src in tqdm(ts_data.unique("linksto").get_column("linksto").to_list()):
        feature_map[src] = sorted(
            ts_data.filter(pl.col("linksto") == src)
            .unique("label")
            .get_column("label")
            .to_list()
        )
        freq[src] = vitals_freq if src == "vitalsign" else lab_freq
    return feature_map, freq


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
    write_data = True
    ts_data_list = []
    skipped_due_to_event_count = False
    skipped_due_to_elapsed_time = False

    for events_by_src in pt_events.partition_by("linksto"):
        src = events_by_src.select(pl.first("linksto")).item()
        timeseries = convert_events_to_timeseries(events_by_src)

        if not _validate_event_count(timeseries, min_events, max_events):
            skipped_due_to_event_count = True
            return False, [], skipped_due_to_event_count, False

        timeseries = _handle_missing_features(timeseries, feature_map[src])
        timeseries, ehr_static = _impute_missing_values(timeseries, ehr_static, impute)

        if include_dyn_mean:
            ehr_static = _add_dynamic_mean(timeseries, ehr_static)

        if not no_resample:
            timeseries = _resample_timeseries(timeseries, freq[src])

        if max_elapsed is not None:
            timeseries = add_time_elapsed_to_events(timeseries, edregtime)
            if timeseries.filter(pl.col("elapsed") <= max_elapsed).shape[0] == 0:
                skipped_due_to_elapsed_time = True
                return False, [], False, skipped_due_to_elapsed_time

        ts_data_list.append(timeseries.select(feature_map[src]))

    return (
        write_data,
        ts_data_list,
        skipped_due_to_event_count,
        skipped_due_to_elapsed_time,
    )


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
    return min_events <= timeseries.shape[0] <= max_events


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
    missing_cols = [x for x in features if x not in timeseries.columns]
    return timeseries.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(c) for c in missing_cols]
    )


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
    if impute == "mask":
        timeseries = timeseries.with_columns(
            [pl.col(f).is_null().alias(f + "_isna") for f in timeseries.columns]
        )
        ehr_static = ehr_static.with_columns(
            [pl.col(f).is_null().alias(f + "_isna") for f in ehr_static.columns]
        )
    elif impute in ["forward", "backward"]:
        timeseries = timeseries.fill_null(strategy=impute).fill_null(value=-1)
        ehr_static = ehr_static.fill_null(value=-1)
    elif impute == "value":
        timeseries = timeseries.fill_null(value=-1)
        ehr_static = ehr_static.fill_null(value=-1)
    return timeseries, ehr_static


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
    timeseries_mean = (
        timeseries.drop(["charttime", "linksto"]).mean().with_columns(pl.all().round(3))
    )
    return ehr_static.hstack(timeseries_mean)


def _resample_timeseries(timeseries: pl.DataFrame, freq: str = "1h") -> pl.DataFrame:
    """
    Resample the time-series data to a specified frequency.

    Args:
        timeseries (pl.DataFrame): The input time-series data.
        freq (str): The frequency for resampling (e.g., "1h").

    Returns:
        pl.DataFrame: The resampled time-series data.
    """
    timeseries = timeseries.upsample(time_column="charttime", every="1m")
    # Exclude -1 values before aggregation
    timeseries = timeseries.with_columns([
        pl.when(pl.col(col) == -1).then(None).otherwise(pl.col(col)).alias(col)
        for col in timeseries.columns if timeseries.schema[col] == pl.Float64
    ])
    # Forward fill null values
    timeseries = timeseries.group_by_dynamic("charttime", every=freq).agg(pl.col(pl.Float64).mean()).fill_null(strategy="forward")
    # Replace remaining null values with -1
    timeseries = timeseries.fill_null(value=-1)
    return timeseries


def _standardize_data(ts_data: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize the 'value' column in the time-series data using min-max scaling.

    Args:
        ts_data (pl.DataFrame): The input time-series data.

    Returns:
        pl.DataFrame: Standardized time-series data.
    """
    min_val = ts_data["value"].min()
    max_val = ts_data["value"].max()
    ts_data = ts_data.with_columns(
        ((pl.col("value") - min_val) / (max_val - min_val)).alias("value")
    )

    return ts_data


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
    print(f"Successfully processed time-series intervals for {n} patients.")
    print(
        f"Skipping {filter_by_nb_events} patients with less or greater number of events than specified."
    )
    print(
        f"Skipping {missing_event_src} patients due to at least one missing time-series source."
    )
    print(
        f"Skipping {filter_by_elapsed_time} patients due to no measures within elapsed time."
    )
