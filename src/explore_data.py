import argparse
import os
import sys

import polars as pl
import toml
import utils.exploration as m4exp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data exploration functionality for MIMIC-IV v3.1."
    )
    parser.add_argument(
        "ehr_path",
        type=str,
        help="Directory containing pre-extracted MIMIC-IV EHR data.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to config toml file containing lookup fields for grouping.",
        default="../config/targets.toml",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        help="Directory where summary tables and plots should be written.",
        required=True,
    )
    parser.add_argument(
        "--display_dict_path",
        "-d",
        type=str,
        help="Path to dictionary for display names of features.",
        default="../config/feat_name_map.json",
    )
    parser.add_argument(
        "--pval_adjust",
        "-p",
        type=str,
        help="Method for p-value adjustment in summary tables and distribution plots. Defaults to 'bonferroni'.",
        default="bonferroni",
    )
    parser.add_argument(
        "--pval_test",
        type=str,
        help="Test type for comparing distribution of BHC token lengths. Defaults to 't-test welch'.",
        default="t-test_welch",
    )
    parser.add_argument(
        "--max_i",
        "-i",
        type=int,
        help="Number of rows argument for plotting function. Defaults to 2.",
        default=2,
    )
    parser.add_argument(
        "--max_j",
        "-j",
        type=int,
        help="Number of columns argument for plotting function. Defaults to 2.",
        default=2,
    )
    parser.add_argument(
        "--rot",
        "-r",
        type=int,
        help="Rotation degrees for ticks in plotting function. Defaults to 0.",
        default=0,
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Whether to use lazy mode for reading in data. Defaults to False.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity.")

    args, _ = parser.parse_known_args()
    if not os.path.exists(args.ehr_path):
        print(f"Path to EHR data {args.ehr_path} does not exist.")
        sys.exit(1)

    ed_pts = pl.read_csv(os.path.join(args.ehr_path, "ehr_static.csv"))
    ed_measures = pl.read_csv(os.path.join(args.ehr_path, "events_ts.csv"))
    if args.lazy:
        ed_pts = ed_pts.lazy()
        ed_measures = ed_measures.lazy()

    if not os.path.exists(args.output_path):
        print(f"Creating output directory for data exploration at {args.output_path}")
        os.makedirs(args.output_path)

    config = toml.load(args.config)
    outcomes = config["outcomes"]["labels"]
    outcome_labels = config["outcomes"]["display"]
    attributes = config["attributes"]["labels"]
    attribute_labels = config["attributes"]["display"]
    nn_attr = config["attributes"]["nonnormal"]
    categorical = config["attributes"]["categorical"]
    age_bins = config["age"]["bins"]
    age_labels = config["age"]["labels"]

    if len(categorical) == 0:
        categorical = None

    print(
        "Generating summary tables and plots across all defined outcomes and attributes."
    )
    ed_pts = m4exp.assign_age_groups(
        ed_pts, bins=age_bins, labels=age_labels, use_lazy=args.lazy
    )
    for outcome, label in zip(outcomes, outcome_labels, strict=False):
        if outcome == 'outcome_AcuteMI':
            continue
        print(f"Generating summary table one by {label}..")
        m4exp.get_table_one(
            ed_pts,
            ed_measures,
            outcome,
            label,
            args.output_path,
            args.display_dict_path,
            sensitive_attr_list=attribute_labels,
            nn_attr=nn_attr,
            verbose=args.verbose,
            adjust_method=args.pval_adjust,
            cat_cols=categorical,
        )
