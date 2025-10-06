# Getting Started

## Data Access
This project makes use of four main modules linked to the MIMIC-IV dataset:

- _hosp_: measurements recorded during hospital stay for training, including demographics, lab tests, prescriptions, diagnoses and care provider orders
- _ed_: records metadata during ED attendance in an externally linked database
- _icu_: records individuals with associated ICU admission during the episode with additional metadata (used mainly for measuring the ICU admission outcome)
- _note_: records deidentified discharge summaries as long form narratives which describe reason for admission and relevant hospital events

To get started, you will need to install the repository and download the [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) data files from PhysioNet (version 3.1). This also includes the [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/) dataset (_ed_ module, v2.2), the [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) (_note_ module, v2.2) and the linked [MIMIC-IV-Ext-BHC](https://physionet.org/content/labelled-notes-hospital-course/1.2.0/) segments that include the Brief Hospital Course summary from the discharge notes (v1.2). The scripts will assume that you have data access to the _hosp_, _icu_, _ed_ and _note_ modules. These should be stored using the following folder structure, matching each data source:
```sh
MIMIC-IV/
├── mimic-iv-ed/3.1/
│   ├── ed/
│   ├── icu/
├── mimic-iv/3.1/
│   └── hosp/
├── mimic-iv-note/3.1/
│   ├── note/
│   ├── mimic-iv-bhc.csv ### Brief Hospital Course segments
mm-healthfair/
```
## Installation
Refer to [README](https://github.com/nhsengland/mm-healthfair/tree/main) for installation instructions. Recommended to use `poetry` to avoid compatibility issues.

## Running the scripts
All scripts are best executed via the command-line in order to specify the required and optional arguments.
- If you have created your own virtual environment with dependencies specified by `requirements.txt` then you should run scripts with:

```sh
source activate venv
python3 src/any_script.py [required args] --[optional args]
```

- If using `poetry` (recommended):
```sh
cd mm-healthfair
poetry run python3 src/any_script.py [required args] --[optional args]
```

## Data Curation
### Downloading the data
The MIMIC-IV dataset (v3.1) can be downloaded from [PhysioNet](https://physionet.org).

Steps to download:

1. Create an account on [PhysioNet](https://physionet.org).
2. Complete the required training and credentialisation.
    - Whilst MIMIC-IV is an open dataset, training and credentialisation is required to access and download the zip files. Please refer to the [PhysioNet](https://physionet.org) webpages for instructions on how to gain access.
3. Download the data.


### Extracting the data
The `extract_data.py` module uses the base **MIMIC-IV** database and the **MIMIC-IV-Ext-BHC** extension as raw data inputs. This feeds into a series of sequential operations for extracting and cleaning the individual data modalities. The data sources used for each modality include:
- **Tabular (Static EHR) data modality**: demographics, prescriptions, previous diagnoses and provider orders (MIMIC-IV)
- **Time-series data modality**: ED vital signs, in-hospital lab tests and measurements (MIMIC-IV and MIMIC-IV-ED)
- **Text modality**: BHC segments from clinical notes (MIMIC-IV-Note and MIMIC-IV-Ext-BHC)

This script will also access the `lab_items.txt` (50 most common lab test IDs), `icd9to10.txt` (ICD-9 to ICD-10 diagnosis mapping table, provided by [Gupta](https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/tree/main) et al.) and `ltc_mapping.json` (ICD-10 coding to long-term condition groups for measuring multimorbidity).

```sh
extract_data.py [-h] --output_path OUTPUT_PATH [--event_tables EVENT_TABLES [EVENT_TABLES ...]]
                    [--include_notes] [--labitems LABITEMS] [--verbose] [--sample SAMPLE]
                    [--lazy] mimic4_path
```

- `mimic4_path`: [Required] Directory containing downloaded MIMIC-IV data.
- `--output_path`: Directory where per-subject data should be written.
- `--include_notes` or `-n`: Extract free-text modality tables.
- `--include_events` or `-t`: Extract time-series modality tables.
- `--include_addon_ehr_data` or `-a`: Extract provider order and prescribing tabular data.
- `--labitems`: Text file containing list of ITEMIDs to use from labevents.
- `--icd9_to_icd10`: Text file containing ICD 9-10 mapping.
- `--ltc_mapping`: JSON file containing mapping for long-term conditions in ICD-10 format.
- `--verbose`: Control verbosity. If true, will make more .collect() calls to compute dataset size.
- `--sample`: Extract randomized subset of patients for testing.
- `--top_n_meds`: Number of drug-level features to extract (if using add-on EHR data).
- `--lazy`: Whether to use lazy mode for reading in data. Defaults to False (except for event tables - always uses lazymode).

Example:

```sh
poetry run python extract_data.py ../../data/MIMIC-IV -o ../outputs/ext_data -n -t -a --top_n_meds 50 -v --lazy
```

This command will read from the `../../data/MIMIC-IV` directory and extract the relevant measurements across the three data modalities for the complete population with linked ED attendances. The script will also create count features covering the top 50 most common prescriptions. ed/vitalsign.csv for 1000 stays. Three output files describing each data modality (`ehr_static.csv`, `events_ts.csv` and `notes.csv`) will be saved under the `../outputs/ext_data` folder.

### Data Exploration

Optionally, the `explore_data.py` script then allows the user to run basic data exploration on the three data files, generating a cohort summary describing the patient cohorts by outcome and a series of distribution plots by attribute. This script will access the `targets.toml` config file for the attribute and outcome names and the `feat_name_map.json` mapping for looking up the display names for each feature.

Usage:
```sh
usage: explore_data.py [-h] [--config CONFIG] --output_path OUTPUT_PATH
                       [--display_dict_path DISPLAY_DICT_PATH] [--bhc_fname BHC_FNAME]
                       [--pval_adjust PVAL_ADJUST] [--pval_test PVAL_TEST] [--max_i MAX_I] [--max_j MAX_J]
                       [--rot ROT] [--lazy] [--verbose]
                       ehr_path
```

- `ehr_path`: [Required] Path to folder containing extracted data.
- `--config`: Path to targets.toml file containing lookup fields for grouping.
- `--output_path`: Directory where summary tables and plots should be written.
- `--display_dict_path`: Path to dictionary for display names of features.
- `--bhc_fname`: Output file name for Brief Hospital Course distribution plot.
- `--pval_adjust`: Method for p-value adjustment in summary tables and distribution plots. Defaults to 'bonferroni'.
- `--pval_test`: Statistical test type for comparing distribution of BHC token lengths. Defaults to 't-test welch'.
- `--max_i`: Number of rows argument for plotting function. Defaults to 2.
- `--max_j`: Number of columns argument for plotting function. Defaults to 2.
- `--rot` or `-r`: Rotation degrees for ticks in plotting function. Defaults to 0.

Example:

```sh
poetry run python explore_data.py ../outputs/ext_data/ -c ../config/targets.toml -o ../outputs/exp_data --lazy -v -r 35
```

This will read the extracted MIMIC-IV files from `../outputs/ext_data/` and the outcome/attribute configurations `../config/targets.toml` and generate the following tables and plots:
- Tableone style summaries of some of the tabular features grouped by outcome.
- Barcharts highlighting outcome prevalence.
- Age group distributions by each sensitive attribute.
- Violin plots of the log-transformed **Brief Hospital Course** lengths with statistical testing.

### Data Preprocessing
Once these core files have been extracted, we run `prepare_data.py` to create a multimodal feature dictionary for model training and evaluation. The time-series and text modalities can optionally be left out of the dictionary. This will execute a sequence of preprocessing strategies for each data modality in the following order:

```sh
├── Tabular modality
│   ├── Encode categorical features
│   ├── Extract lookup date fields
│   ├── Remove correlated features
├── Time-series modality
│   └── Generate measurement intervals (vitals and lab tests)
│       ├──── Filter patients with valid measurements
│       ├──── Feature standardisation
│       ├──── Set sampling frequency (from targets.toml)
│       ├──── Data imputation
├── Text modality
│   ├── Clean redacted segments
│   ├── Generate BioBERT sentence-level embeddings
│   ├── Append embeddings to feature dictionary
├── Training-Validation-Test split
├── Export feature dictionary
```

Usage:
```sh
prepare_data.py [-h] [--output_dir OUTPUT_DIR] [--output_summary_dir OUTPUT_SUMMARY_DIR]
                       [--output_reference_dir OUTPUT_REFERENCE_DIR] [--pkl_fname PKL_FNAME]
                       [--col_fname COL_FNAME] [--config CONFIG] [--corr_threshold CORR_THRESHOLD]
                       [--corr_method CORR_METHOD] [--min_events MIN_EVENTS] [--max_events MAX_EVENTS]
                       [--impute IMPUTE] [--no_resample] [--include_dyn_mean] [--standardize]
                       [--max_elapsed MAX_ELAPSED] [--include_notes] [--train_ratio TRAIN_RATIO] [--stratify]
                       [--seed SEED] [--verbose]
                       data_dir
```


- `data_dir`: [Required] Path to folder containing extracted data.
- `--output_dir`: Directory to save processed training ids and pkl files.
- `--output_summary_dir`: Directory to save summary table file for training/val/test split.
- `--output_reference_dir`: Directory to save lookup table file containing dates and additional fields from EHR data.
- `--pkl_fname`: Name of pickle file to save processed data.
- `--col_fname`: Name of pickle file to save column lookup dictionary.
- `--config`: Path to targets.toml file containing lookup fields for grouping.
- `--corr_threshold`: Threshold for removing correlated features. Features with correlation above this value will be removed.
- `--corr_method`: Method for removing correlated features. Defaults to Pearson correlation.
- `--min_events`: Minimum number of events per patient (time-series modality).
- `--max_events`: Maximum number of events per patient.
- `--impute`: Impute strategy. One of ['forward', 'backward', 'mask', 'value' or None].
- `--no_resample`: Flag to turn off time-series resampling.
- `--include_dyn_mean`: Flag for whether to add mean of dynamic features to static data.
- `--standardize`: Flag for whether to standardize timeseries data with minmax scaling (in the range [0,1]).
- `--max_elapsed`: Max time elapsed from hospital admission (hours). Filters any events that occur after this.
- `--include_notes`: Whether to preprocess notes if available using BioBERT tokenization.
- `--train_ratio`: Ratio of training data to create split.
- `--stratify`: Whether to stratify the split by outcome and sensitive attributes.
- `--seed`: Seed for random sampling. Defaults to 0.

Example:

```sh
poetry run python prepare_data.py ../outputs/ext_data -o ../outputs/prep_data --train_ratio 0.8 --min_events 2 --impute value --max_elapsed 72 --stratify --include_notes -v
```

This will initiate the complete multimodal feature extraction across the three modalities. The target data split will retain 80% of unique patients for training, 10% for validation and 10% for testing, stratified by the target outcome. The patient set will be further restricted to having at least 2 measurements across lab and vital measurements, collected within 72 hours of ED arrival. Data imputation will simply replace the missing time-series measurements with '-1'. The text pipeline will use SpaCy to automatically download the [BioBERT](https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT) pre-trained discharge summary embeddings and execute a PyTorch embedding pipeline to tokenize the data inputs. It's worth noting that this process may take several hours, if running on a local machine.

The final dictionary will be exported to the specified folder under `../outputs/prep_data/<--pkl_fname>.pkl`, where each primary key will be a unique patient, and each data modality and target outcome set will be separate sub-keys within each patient sample. The column names for each field will be stored in a separate dictionary with the same data structure in `../outputs/prep_data/<--col_fname>.pkl`. The training/val/test ids will be stored in separate `.csv` files inside `--output_dir`.

## Model Development and Evaluation

### Multimodal Learning

To train a model, we use `train.py` to generate a customisable deep fusion model for risk classification. To execute the pipeline, we require the multimodal feature dictionary, the columns dictionary and the train/val/test id files, acquired from the previous step. The pipeline currently supports intermediate fusion across tabular, time-series and free-text modalities. This also includes the ability to perform adversarial mitigation on select attribute groups within the tabular data (gender, ethnicity, marital status and insurance). Each modality is fed through a stand-alone deep learning component and fused at the final hidden layer through concatenation. The deep neural net components are as follows:
- Static (EHR) modality: MLP classifer (EHR-MLP).
- Time-series modality: LSTM classifier (TS-LSTM).
- Text modality: Transformer-encoder classifier (NT-TF-E).
- Adversarial heads: Deep Multi-unit Adversarial classifier if using debiasing.
- Tested multimodal combinations: Static+TS (IF-EHR+TS) and Static+TS+Text (IF-EHR+TS+NT).

Usage:

```sh
train.py [-h] [--col_path COL_PATH] [--ids_path IDS_PATH] [--outcome OUTCOME] [--config CONFIG]
                [--targets TARGETS] [--cpu] [--sample SAMPLE] [--use_class_weights] [--use_debiasing]
                [--project PROJECT] [--wandb]
                data_path
```

- `data_path`: [Required] Path to the processed data .pkl file.
- `--col_path`: Path to the pickled column dictionary generated from prepare_data.py.
- `--ids_path`: Directory containing train/val/test ids.
- `--outcome`: Binary outcome to use for multimodal learning (one of the labels in targets.toml). Defaults to prediction of in-hospital death.
- `--config`: Path to `model.toml` file containing model training parameters.
- `--targets`: Path to `targets.toml` file containing lookup fields and outcomes.
- `--cpu`: Whether to use CPU compute for training. Defaults to gpu.
- `--sample`: Randomly sample subjects for testing. Training set will equal sample, validation set will be 1/5th of sample.
- `--use_class_weights`: Whether to use class weights for imbalanced classification tasks. Defaults to False.
- `--use_debiasing`: Whether to use adversarial debiasing to penalize the model's decisions if influenced by sensitive attributes. Target attribute indices in dictionary can be defined in dbindices within targets.toml. Defaults to False.
- `--project`: Name of project, used for wandb logging.
- `--wandb`: Whether to use wandb for logging. Defaults to False.

**Note**: If `--wandb` is set, then we use the Weights&Biases service for logging the experiment and storing the model artifacts. You would first need to create a free account and will be prompted to login on the first run and setup and access token. You would also need to set the `--project` name to match the name of the W&B workspace. Refer to the [wandb + lightning docs](https://docs.wandb.ai/guides/integrations/lightning#sign-up-and-log-in-to-wandb) for more details. After finishing the run, you can download the coded model artifact (displayed during execution) and track validation set performance within W&B.

Example:

```sh
poetry run python train.py ../outputs/prep_data/mmfair_feat.pkl -p ../outputs/prep_data/mmfair_cols.pkl -o ext_stay_7 --project nhs-mm-healthfair --wandb --use_debiasing
```

This will run the multimodal training pipeline with enabled logging to W&B for prediction of extended hospital stay (>=7 days). The model hyperparameters are configurable from the `models.toml` file including the fusion method (`None`, `concat` or `mag`). The target modalities need to be explicitly set within the `modalities` list in the config file (the values need to be one or more of `static`, `timeseries` or `notes`). Finally, we can also customize the training batch size, early stopping, learning rate and the strength parameter for debiasing.

**Note:** The `st_first` parameter can be customised to fuse the tabular or the time-series data first if using the `mag` fusion method (multi-adaptation gates). To test a model with time-series as the primary modality, you should set `st_first=False` in the `model.toml` file (otherwise uses default value of True). Note that this has only been tested in the previous project iteration using these two modalities.

If using debiasing, you can customise the objective for **adversarial mitigation** objective by editing the `dbindices` list within `targets.toml`. For each feature index the pipeline will create an adversarial head unit for classification. Then, the adversarial loss function will be fed through a Gradient Reversal Layer to reverse the objective for optimisation (maximising the loss to reduce biases, while minimising the main objective to predict the outcome). The strength of debiasing can be customised using the `adv_lambda` parameter (tested within the range [0, 5]).

### Multimodal Evaluation
Once you have trained and saved a model you can run inference on the hold-out test set with `evaluate.py` to display a performance summary with confidence intervals. This also includes the ability to perform risk stratification by binning the output probabilities in equally-sized quantiles. We can then use the `--strat_by_attr` parameter to generate plots describing the risk quantiles by each sensitive attribute specified in `targets.toml`. The script supports two separate evaluation modes:
- **Single mode** (`group_models=False`): Evaluates a single learner generating independent ROC/PR/Calibration/Loss curves and risk stratification summary that includes ranking the prediction quantiles by attribute.
- **Group mode** (`group_models=True`): Group evaluation, where the script will generate grouped ROC and PR curves over a list of saved model files (model names must be specified in the `targets.toml` paths section.

Usage:
```sh
 evaluate.py [-h] [--col_path COL_PATH] [--ids_path IDS_PATH] [--attr_path ATTR_PATH]
                   [--eval_path EVAL_PATH] [--outcome OUTCOME] [--model_dir MODEL_DIR]
                   [--model_path MODEL_PATH] [--config CONFIG] [--targets TARGETS] [--n_bins N_BINS]
                   [--strat_by_attr] [--group_models] [--verbose]
                   data_path
```

- `data_path`: [Required] Path to the processed data .pkl file.
- `--col_path`: Path to the pickled column dictionary generated from `prepare_data.py`.
- `--ids_path`: Directory containing the test set ids.
- `--attr_path`: Directory containing attributes metadata (original tabular dataset `ehr_static.csv`).
- `--eval_path` or `-e`: Directory to store performance plots.
- `--outcome` or `-o`: Binary outcome to use for multimodal learning (one of the labels in `targets.toml`). Defaults to prediction of in-hospital death.
- `--model_dir` or `-m`: Directory containing the saved model metadata.
- `--model_path`: Directory pointing to the saved model checkpoint (must be inside model_dir).
- `--config`: Path to `model.toml` file containing model training parameters.
- `--targets`: Path to `targets.toml` file containing lookup fields and outcomes.
- `--n_bins`: Number of bins for quantile analysis and calibration curves.
- `--strat_by_attr`: Show stratified quantile analysis for risk stratification grouped by attribute.
- `--group_models`: Group evaluation mode. Generate Precision-Recall summary across all models specified in `targets.toml`.

To execute the inference we require the multimodal feature dictionary generated from the previous step, as well as the column mapping dictionary and the original tabular data file (for extracting the unprocessed sensitive attributes). The script will print out a performance summary with confidence intervals and save a series of performance plots to `eval_path`. It will also store a separate dictionary file in `<eval_path>/pf_<model_path>.pkl` which will contain the metric values and risk quantile mappings per patient. This dictionary is be required in the following explainability analysis.

Example:

**Single mode**
```sh
poetry run python evaluate.py ../outputs/prep_data/mmfair_feat.pkl -e ../outputs/evaluation -o ext_stay_7 -m ext_stay_7_concat_static_timeseries_notes -v --strat_by_attr
```
This runs inference on a multimodal algorithm fusing tabular, timeseries and text modalities for prediction of extended hospital stay. The model checkpoint file must be located in the `-m` directory. We additionally include the option to risk stratify the predictions and generate stacked barcharts of the attribute distributions across each risk quantile.

**Group mode**
```sh
poetry run python evaluate.py ../outputs/prep_data/mmfair_feat.pkl -e ../outputs/evaluation -o ext_stay_7 --group_models -v
```
This runs inference on a list of multimodal algorithms, with each model directory, display name and color for plotting explicitly set in `targets.toml` and its `paths` section.

### Fairness Evaluation

In similar fashion, we can run `fairness.py` to evaluate a multimodal algorithm based on fairness metrics (Demographic Parity, Equalised Odds and Equal Opportunity). The script will use BCa bootstrapping to generate 95% CIs across each fairness metrics, over a set number of bootstrap iterations. There are two options for thresholding the output probabilities, selected using the `--threshold_method`. These are either Youden's J-statistic (default), selecting the best cutoff maximising the distance between TPR and FPR, or Max F1-score, optimised for precision and TPR strictly on the positive class. The [FairLearn](https://fairlearn.org) package is used to compute the fairness metrics across the sensitive attributes and perform error analysis. Once again, we can execute the script in two modes:
- **Single mode** (`across_models=False`): Evaluates fairness for a single learner generating bootstrapped fairness metrics, error plots and age-stratified fairness plots. This will save a fairness dictionary under `<fair_path>/pf_<model_path>.pkl` storing the computed fairness values over the set bootstraps. Group mode will then require this dictionary for each target model to generate a grouped fairness summary.
- **Group mode** (`across_models=True`): Group evaluation, where the script will generate a grouped barplot with confidence intervals across each model that has been specified in `targets.toml` and its `paths` section (same location as in `evaluate.py`).

Usage:
```sh
 evaluate.py [-h] [--col_path COL_PATH] [--ids_path IDS_PATH] [--attr_path ATTR_PATH]
                   [--eval_path EVAL_PATH] [--outcome OUTCOME] [--model_dir MODEL_DIR]
                   [--model_path MODEL_PATH] [--config CONFIG] [--targets TARGETS] [--n_bins N_BINS]
                   [--strat_by_attr] [--group_models] [--verbose]
                   data_path
```

- `data_path`: [Required] Path to the processed data .pkl file.
- `--fair_path` or `-f`: Directory to store fairness plots.
- `--model_dir`: Directory containing the saved model metadata.
- `--model_path`: Directory pointing to the saved model checkpoint (must be inside model_dir).
- `--attr_path`: Directory containing attributes metadata (original ehr_static.csv).
- `--outcome` or `-o`: Binary outcome to use for multimodal evaluation (one of the labels in targets.toml). Defaults to prediction of extended hospital stay.
- `--targets`: Path to targets.toml file containing lookup fields and outcomes.
- `--group_by_age`: If true, will generate a trajectory lineplot of the fairness measures across each age group (as assigned in targets.toml).
- `--plot_errors`: If true, will use the Fairlearn API to generate a grouped barplot of error measures over each attribute (as specified in targets.toml).
- `--across_models`: If true, will generate a trajectory lineplot of the fairness measures across multiple models (specified in paths within targets.toml).
- `--threshold_method`: Method to use for thresholding positive and negative classes under class imbalance. Options are 'yd' (Youden's J statistic) or 'f1' (Maximum achievable F1-score). Defaults to 'yd'.
- `--boot_samples` or `-b`: Number of bootstrap samples to use for calculating confidence intervals.
- `--seed`: Seed for bootstrapping reproducibility. Defaults to 42.

Example:

**Single mode**
```sh
poetry run python fairness.py ../outputs/evaluation -f ../outputs/fairness -o ext_stay_7 -m ext_stay_7_concat_static_timeseries_notes -b 1000 --plot_errors -v
```
This runs inference on a multimodal algorithm fusing tabular, timeseries and text modalities for prediction of extended hospital stay. In this setup, the BCa will estimate the DPR, EQO and EOP metrics over 1000 bootstraps within the testing set. It will then use the Fairlearn API to generate error plots, print out the fairness metrics and store the values in a fairness dictionary within the target folder `-f`.

**Group mode**
```sh
poetry run python fairness.py ../outputs/evaluation -f ../outputs/fairness -o ext_stay_7 --across_models
```
This runs fairness inference on a list of multimodal algorithms, with each model directory, display name and color for plotting explicitly set in `targets.toml` and its `paths` section. The output will be a grouped barchart by sensitive attribute showing the global differences in DPR, EQO and EOP.

**Note**: This requires each model specified in `targets.toml` to have been run in single mode first and have its fairness dictionary stored in their respective fairness output folder.

### Explainability Analysis

Finally, we can run `explain.py` to measure feature importance and understand the decision boundaries of a multimodal algorithm. Currently the script only supports interpreting a fully-fused model across all three data modalities (IF-EHR+TS+NT). It uses the [SHAP](https://shap.readthedocs.io/en/latest/) library for post-model global and local feature importance estimation and borrows ideas from the [MM-SHAP](https://github.com/Heidelberg-NLP/MM-SHAP/tree/main) library for aggregating SHAP values across modalities.

Upon its first run, the script will create a model wrapper using the **DeepSHAP** algorithm to backpropagate across the multimodal framework and estimate the Shapley values across each deep neural net component. Similar to the previous scripts, it will again save a `.pkl` dictionary file containing the batch-wise Shapley values, mapped by each patient ID in the testing set, and split across each data modality. The time-series modality will include two independent value sets for the vital signs and the lab testing measurements. This process may take ~30 minutes on a local machine. After storing the SHAP values, the dictionary containing the attribution scores will be pre-loaded automatically in future runs. The next step of the script will depend on the selected mode:

- **Global mode** (`exp_mode=global`): This will plot grouped SHAP summary plots (or heatmap plots if `--use_heatmaps` is specified) highlighting feature importances for each modality.
- **Local mode** (`exp_mode=local`): This will plot individual-level SHAP decision plots, including a text highlight plot to highlight important regions within the clinical note segments. The individual can be randomly sampled using `--local_risk_group` to select the target risk quantile (estimated with `evaluate.py`) and a target patient profile selecting gender, marital status, insurance and ethnicity attributes in `targets.toml` within the `shap_profile` section.

Usage:
```sh
 explain.py [-h] [--col_path COL_PATH] [--feat_names FEAT_NAMES] [--ids_path IDS_PATH]
                  [--exp_path EXP_PATH] [--eval_path EVAL_PATH] [--attr_path ATTR_PATH] [--outcome OUTCOME]
                  [--model_dir MODEL_DIR] [--model_path MODEL_PATH] [--config CONFIG] [--targets TARGETS]
                  [--exp_mode EXP_MODE] [--use_heatmaps] [--global_max_features GLOBAL_MAX_FEATURES]
                  [--local_risk_group LOCAL_RISK_GROUP] [--notes_offset_ref] [--verbose]
                  data_path
```

- `data_path`: [Required] Path to the processed data .pkl file.
- `col_path`: Path to the pickled column dictionary generated from `prepare_data.py`.
- `--feat_names`: Path to a JSON file containing lookup names for each feature.
- `--ids_path`: Directory containing test set ids.
- `--exp_path` or `-x`: Directory to store explanation plots.
- `--eval_path` or `-e`: Evaluation path for obtaining risk dictionary as .pkl file.
- `--attr_path`: Directory containing attributes metadata (original `ehr_static.csv`).
- `--outcome` or `-o`: Binary outcome to use for multimodal learning (one of the labels in `targets.toml`). Defaults to prediction of in-hospital death.
- `--model_dir`: Directory containing the saved model metadata.
- `--model_path`: Directory pointing to the saved model checkpoint (must be inside `model_dir`).
- `--config`: Path to `model.toml` file containing model training parameters.
- `--targets`: Path to `targets.toml` file containing lookup fields and outcomes.
- `--exp_mode`: Use global or local explanations. If global, uses DeepExplainer over static and timeseries modalities in the test set. If local, uses DeepExplainer to explain individual predictions over all modalities (requires a fused static-timeseries-text model).
- `--use_heatmaps`: Use heatmaps for SHAP summary plots instead of beeswarm plots.
- `--global_max_features`: Top N features to plot in global-level explanations.
- `--local_risk_group`: Risk quantile to use for local-level explanations (generated in evaluate.py).
- `--notes_offset_ref`: Offset SHAP colormap center for local-level text plot using the expected SHAP value (batch-wise mean). Defaults to False (SHAP colormap centered around 0).

Example:

**Global mode**
```sh
poetry run python explain.py ../outputs/prep_data/mmfair_feat.pkl -m ext_stay_7_concat_static_timeseries_notes_db -o ext_stay_7 -x ../outputs/explanations --exp_mode global --global_max_features 20 --use_heatmaps -v
```
This runs explainability inference in global mode on the fully-fused multimodal algorithm for prediction of extended hospital stay. In this setup, the script will aggregate all Shapley values across data batches, per modality. This will be used to generate global summary plots describing the most influential features within each modality. If not using heatmaps, the output will be a series of SHAP density plots highlighting the patient distributions with respect to the computed importance scores. If using heatmaps (`--use_heatmaps`), the [SHAP heatmap](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/heatmap.html) plot will be used to group examples by their explanation similarity (via hierarchical clustering). This results in samples that have the same model output for the same reason getting grouped together and sorted based on the absolute mean value of the feature-wise SHAP scores. The text modality uses an alternative interface, which is the [SHAP barplot](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html) showcasing the most influential note segments across all data batches in the test set, regardless of risk direction.

**Local mode**
```sh
poetry run python explain.py ../outputs/prep_data/mmfair_feat.pkl -m ext_stay_7_concat_static_timeseries_notes -o ext_stay_7 -x ../outputs/explanations --exp_mode local -r 10 -v
```
This runs explainability inference in local mode on the fully-fused multimodal algorithm for prediction of extended hospital stay. This means that a randomly sampled individual from risk quantile 10 (highest-risk group) will be selected based on the patient profile specified in `targets.toml` within `[shap_profile]`. The script will load in the pre-computed SHAP dictionary from `../outputs/explanations/ext_stay_7_concat_static_timeseries_notes` and retrieve the batch where the target patient ID is stored. Then, a series of Explanation objects will be created, with assigned background (reference) values equal to the batch mean Shapley values for each modality. These values will be treated as the baseline for observing changes in risk based on feature impact within each modality interface. The final plot contains four interfaces: three decision plots highlighting change in risk direction across the top ranked features within the tabular, time-series (vitals) and time-series (lab tests) modalities and one text plot highlighting important segments within the Brief Hospital Course discharge summaries. The [MM-SHAP](https://github.com/Heidelberg-NLP/MM-SHAP/tree/main) scores are also computed to display relative degrees of modality dependence on an individual-level model decision.

**Note**: This requires each model specified in `targets.toml` to have been run in single mode first and have its fairness dictionary stored in their respective fairness output folder.

## General Usage Tips
- Extending the framework: To implement a new fusion method, create another `LightningModule` subclass in `models.py` and edit `train.py` to use your new model. To implement a custom dataloader for another dataset, you can edit `datasets.py` to define a custom PyTorch Dataset class.
- Multimodal setup considerations: It is generally recommended to keep to a strict and consistent naming convention for the model/fairness/explanation folders, describing the model run. In this case, the model path and file names should follow the convention:
`<outcome>_<fusion_method>_<modalities>`, e.g. `ext_stay_7_concat_static_timeseries_notes`, describing the fully-fused multimodal algorithm for prediction of extended hospital stay.
- Other naming conventions: currently supported outcomes are labelled as [`in_hosp_death`, `ext_stay_7`, `non_home_discharge`, `icu_admission`]; currently supported sensitive data attributes are labelled as [`gender`, `race_group`, `marital_status`, `insurance`].
- Training setup tips: Increase `num_workers` to enable multithreading and speed up the training process. Adjust `learning_rate` and `early_stopping` to improve convergence rate.
