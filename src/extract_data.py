import argparse
import gzip
import os
import shutil
import sys
import warnings

import numpy as np
import polars as pl
import utils.mimiciv as m4c
from utils.functions import (
    get_demographics_summary,
    get_final_episodes,
    get_n_unique_values,
)
from utils.preprocessing import get_diag_features, preproc_icd_module, clean_vitals

warnings.filterwarnings("ignore", category=pl.exceptions.MapWithoutReturnDtypeWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from MIMIC-IV v3.1.")
    parser.add_argument(
        "mimic4_path", type=str, help="Directory containing downloaded MIMIC-IV data."
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        help="Directory where per-subject data should be written.",
        required=True,
    )

    parser.add_argument("--include_events", "-t", default=False, action="store_true")
    parser.add_argument(
        "--include_addon_ehr_data", "-a", default=False, action="store_true"
    )
    parser.add_argument(
        "--labitems",
        "-i",
        type=str,
        help="Text file containing list of ITEMIDs to use from labevents.",
        default="../config/lab_items.txt",
    )
    parser.add_argument(
        "--icd9_to_icd10",
        type=str,
        help="Text file containing ICD 9-10 mapping.",
        default="../config/icd9to10.txt",
    )
    parser.add_argument(
        "--ltc_mapping",
        type=str,
        help="JSON file containing mapping for long-term conditions in ICD-10 format.",
        default="../config/ltc_mapping.json",
    )
    parser.add_argument(
        "--outcome_mapping",
        type=str,
        help="JSON file containing mapping for long-term conditions in ICD-10 format.",
        default="../config/outcome_mapping.json",
    )
    parser.add_argument(
        "--proc_mapping",
        type=str,
        help="JSON file containing mapping for long-term conditions in ICD-10 format.",
        default="../config/proc_mapping.json",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        action="store_true",
        default=False,
        help="Control verbosity. If true, will make more .collect() calls to compute dataset size.",
    )
    parser.add_argument(
        "--sample",
        "-s",
        type=int,
        help="Extract smaller patient sample (random).",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Whether to use lazy mode for reading in data. Defaults to False (except for events tables - always uses lazymode).",
    )

    args, _ = parser.parse_known_args()
    mimic4_path = os.path.join(args.mimic4_path, "mimiciv", "3.1", "hosp")
    mimic4_ed_path = os.path.join(args.mimic4_path, "mimic-iv-ed", "3.1", "ed")
    mimic4_icu_path = os.path.join(args.mimic4_path, "mimic-iv-ed", "3.1", "icu")
    mimic4_ecg_path = os.path.join(args.mimic4_path, "mimic-iv-ecg", "1.0")

    if os.path.exists(args.output_path):
        response = input("Will need to overwrite existing directory... continue? (y/n)")
        if response == "y":
            try:
                shutil.rmtree(args.output_path)  # delete old dir
                os.makedirs(args.output_path)  # make new dir
            except OSError as ex:
                print(ex)
                sys.exit()
        else:
            print("Exiting..")
            sys.exit()
    else:
        print(f"Creating output directory for extracted subjects at {args.output_path}")
        os.makedirs(args.output_path)

    # Read in csv files
    admits = m4c.read_admissions_table(
        mimic4_path, mimic4_ed_path, use_lazy=args.lazy, verbose=args.verbose
    )
    if args.verbose:
        print(
            f"START:\n\tInitial stays with ED attendance: {get_n_unique_values(admits, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(admits)}"
        )
    admits = m4c.read_patients_table(mimic4_path, admits, use_lazy=args.lazy)
    if args.verbose:
        print(
            f"\n\tValidated stays at age>=18: {get_n_unique_values(admits, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(admits)}"
        )
        print("Creating patient-level dataset based on final episodes...")
    # Get relevant ICU data
    admits = m4c.read_icu_table(
        mimic4_icu_path, admits, use_lazy=args.lazy, verbose=args.verbose
    )
    if args.verbose:
        print(
            f"\n\tED Attendances after full data linkage: {get_n_unique_values(admits, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(admits)}"
        )
    # Get patients with ECG readings
    admits = m4c.read_ecg_measurements(
         mimic4_ecg_path, mimic4_ed_path, admits, use_lazy=args.lazy
    )
    if args.verbose:
        print(
            f"\n\tED Attendances with ECG within 3 hours: {get_n_unique_values(admits, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(admits)}"
        )

    admits_last = get_final_episodes(admits)

    ### Optional random sampling to understample subjects
    # sample n subjects (can be used to test/speed up processing)
    if args.sample is not None:
        if args.verbose:
            print(
                f"SELECTING RANDOM SAMPLE OF {args.sample} PATIENTS WITH ED ATTENDANCE."
            )
        # set the seed for reproducibility
        rng = np.random.default_rng(0)
        if isinstance(admits_last, pl.LazyFrame):
            admits_last = admits_last.collect()
        admits_last = admits_last.sample(n=args.sample, seed=0)

    # Process long-term conditions
    diagnoses = m4c.read_diagnoses_table(
        mimic4_path, admits, admits_last, use_lazy=args.lazy, verbose=args.verbose
    )
    if args.verbose:
        print(
            f"DIAGNOSES:\n\tUnique ICD-10 conditions across stays: {get_n_unique_values(diagnoses, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(diagnoses)}"
        )
    diagnoses = preproc_icd_module(
        diagnoses,
        icd_map_path=args.icd9_to_icd10,
        cond_dict_path=args.ltc_mapping,
        verbose=args.verbose,
        use_lazy=args.lazy,
    )
    admits_last = get_diag_features(
        admits_last, diagnoses, cond_dict_path=args.ltc_mapping, use_lazy=args.lazy
    )

    # Process procedure history
    admits_last = m4c.read_procedures_table(
        mimic4_path, admits, admits_last, proc_dict_path=args.proc_mapping,
        use_lazy=args.lazy, verbose=args.verbose
    )
    if args.verbose:
        print(f"Unique patients with prior revascularization procedures: {admits_last.collect().filter(pl.col('prev_revasc')==1).select(pl.col('subject_id').n_unique()).to_series()[0]}")

    # Process target outcomes
    outcomes = m4c.read_outcomes_table(
        mimic4_path, admits_last, use_lazy=args.lazy
    )
    if args.verbose:
        print(
            f"OUTCOMES:\n\tUnique ICD-10 conditions in current episodes: {get_n_unique_values(outcomes, 'hadm_id')}\n\tSUBJECT_IDs: {get_n_unique_values(outcomes)}"
        )
    outcomes = preproc_icd_module(
        outcomes,
        icd_map_path=args.icd9_to_icd10,
        cond_dict_path=args.outcome_mapping,
        outcomes=True,
        verbose=args.verbose,
        use_lazy=args.lazy,
    )
    admits_last = get_diag_features(
        admits_last, outcomes, cond_dict_path=args.outcome_mapping,
        outcomes=True, use_mm=False, use_lazy=args.lazy
    )

    if args.verbose:
        print("------------------------------------------")
        print("Printing characteristics in full patient sample.")
        get_demographics_summary(admits_last)
        print("------------------------------------------")

    if args.include_events:
        print("Getting time-series data from OMR table..")
        omr = m4c.read_omr_table(mimic4_path, admits_last, use_lazy=args.lazy)
        print("Getting time-series data from ED vital signs table..")
        ed_vitals = m4c.read_vitals_table(
            mimic4_ed_path, admits_last, use_lazy=args.lazy
        )
        print("Getting lab test measures..")

        # read compressed and write to file since lazy polars API can only scan uncompressed csv's
        if not os.path.exists(os.path.join(mimic4_path, "labevents.csv")):
            print("Uncompressing labevents data... (required)")
            with gzip.open(os.path.join(mimic4_path, "labevents.csv.gz"), "rb") as f_in:
                with open(os.path.join(mimic4_path, "labevents.csv"), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        labs = m4c.read_labevents_table(
            mimic4_path, admits_last, include_items=True if args.labitems else False, items_path=args.labitems
        )
        print("Merging OMR, ED and Lab test measurements..")
        events = m4c.merge_events_table(ed_vitals, labs, omr, use_lazy=args.lazy)
        print(
            "Cleaning vital signs measurements..."
        )
        events = clean_vitals(events)
        print(
            "Filtering population with ED attendance and measurements history.."
        )

    if args.include_addon_ehr_data:
        print("Parsing additional medication data from the EHR..")
        admits_last = m4c.read_medications_table(
            mimic4_path, admits_last, use_lazy=args.lazy
        )
        if isinstance(admits_last, pl.LazyFrame):
            meds = admits_last.collect(streaming=True).to_pandas()
        else:
            meds = admits_last.to_pandas()
        nums_cols = [col for col in meds.columns if "n_presc" in col]
        meds = meds[nums_cols].sum(axis=1)

        print(
            f"MEDICATIONS (EHR):\n\tParsed medication history with median {meds.median()} administered drugs per patient (IQR: {meds.quantile(0.25)} - {meds.quantile(0.75)})."
        )

    if args.verbose:
        print("Completed data extraction.")
        print("Writing data to disk..")
        if args.include_events:
            m4c.save_multimodal_dataset(
                admits_last, events, output_path=args.output_path
            )
        else:
            m4c.save_multimodal_dataset(
                admits_last,
                admits_last,
                output_path=args.output_path,
            )
        print(f"Exported extracted MIMIC-IV data as CSV files to {args.output_path}.")

    print("Data extraction complete.")
