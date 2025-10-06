# AI-ACC-MIMICIV
Development and validation of a multimodal risk prediction pipeline for diagnosis of acute cardiac conditions using MIMIC-IV.

### About the Project

This repository holds code for developing and testing multimodal fusion approaches for the prediction of acute cardiac conditions.

_**Note:** Only public or fake data are shared in this repository._

### Project Structure

**_Work in progress_**

### Built With

[![Python v3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)

In the latest iteration, the framework was tested locally using [**Python** v3.10.11](https://www.python.org/downloads/release/python-31011/) and tested on a Windows 11 machine with GPU support (NVIDIA GeForce RTX 3080, 16 GiB VRAM).

### Getting Started

#### Installation

**_Work in progress_**

### Usage
This repository contains code used to generate and evaluate multimodal deep learning pipelines for risk prediction using Electronic Health Record data from [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) and ECG waveform data from [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/). In the future, it will include functionalities for adversarial mitigation (controlling model dependence on sensitive attributes), fairness analysis with bootstrapping and explainability using [SHAP](https://shap.readthedocs.io/en/latest/) and [MM-SHAP](https://github.com/Heidelberg-NLP/MM-SHAP/) scores for examining multimodal feature importance.

#### Code outputs
- Preprocessed multimodal features from MIMIC-IV 3.1 and related dictionaries.
- Multimodal learner artifacts (model checkpoints).
- Performance, fairness and explainability summaries mapped by artifact name (coded as `<outcome>_<fusion_type>_<modalities>`, e.g. `ext_stay_7_concat_static_timeseries_notes`).
- Notebooks for debugging, inference relative to the generated dictionary files throughout the pipeline.

#### Datasets
The MIMIC-IV dataset (v3.1) can be downloaded from [PhysioNet.org](https://physionet.org) after completion of mandatory training. This project makes use of four main modules linked to the MIMIC-IV dataset:

- _hosp_: measurements recorded during hospital stay for training, including demographics, lab tests, prescriptions, diagnoses and care provider orders
- _ed_: records metadata during ED attendance in an externally linked database
- _icu_: records individuals with associated ICU admission during the episode with additional metadata (used mainly for measuring the ICU admission outcome)
- _note_: records deidentified discharge summaries as long form narratives which describe reason for admission and relevant hospital events

Additional linked datasets may include [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/) (v2.2), [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) (v1.0) and [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) (v2.1.0). Further information can be found in PhysioNet's [documentation](https://mimic.mit.edu/).

### Roadmap

- [x] Setting up base code repository
- [ ] Configuring project environment and packages
- [ ] Developing feature extraction code for prediction of acute cardiac conditions
- [ ] Developing and testing unimodal deep learning pipelines with ECG/CXR data
- [ ] Developing and testing multimodal deep learning pipelines with fusion of EHR/ECG/CXR data
- [ ] Developing module for performance assessment and calibration
- [ ] Developing module for fairness analytics and healthcare bias assessment
- [ ] Developing module for multimodal explainability and testing feature interactions (e.g. [MM-SHAP](https://github.com/Heidelberg-NLP/MM-SHAP/))

### Contributing

Contributions are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

### Contact and enquiries

Konstantin Georgiev [Konstantin.Georgiev@ed.ac.uk](Konstantin.Georgiev@ed.ac.uk): Insitute of Neuroscience and Cardiovascular Research (University of Edinburgh)
