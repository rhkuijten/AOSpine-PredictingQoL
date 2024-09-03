# AOSpine-PredictingQoL

This repository contains the code and resources used in the project funded by the AO Spine Discovery & Innovation Award (AOS-DIA-22-012-TUM). Our research focused on predicting quality of life (QoL) three months after baseline visit for patients with spinal metastatic disease.

## Overview

The goal of this project was to develop and internally evaluate a prediction model that predicts QoL for patients with spinal metastases across the spectrum of (local) treatment modalities, that uses easily available predictors, and is fair across sociodemographic groups.

## Model web-application
The model can be accessed at our online [web-application](https://spinalmetastases.streamlit.app). The web-application also offers a SHAP waterfall plot explaining your prediction.

## Repository Contents

### Code Overview
1. Pre-processing
- `0_sample_size.py`: Calculates the necessary sample size for training and testing
- `a_EQ5DIndex_complete_data.r`: Calculates EQ-5D-3L index scores using Dutch QoL weights
- `b_EDA_complete_data.py`: Generates a data profile for the complete dataset
- `c_EDA_imputed_data.py`: Generates a data profile for the imputed dataset
- `d_Data_preprocessing.py`: Preprocesses all data to prepare for analysis
- `e_Feature_selection.ipynb`: Jupyter notebook dedicated to feature selection

2. Model
- `precomputed_shap_values.py`: Calculates SHAP values for each potential model input
- `shap_local.ipynb` : Jupyter notebook for testing SHAP waterfall plots
- `testing_fairness.py`: Evaluates the fairness of the final model across different (socio)demographic groups
- `testing_results.py`: Assesses and logs the performance of the final model (discrimination, calibration, Brier score, and decision curve analysis) to Neptune
- `training_baseline.py`: Evaluates the baseline performance (discrimination, calibration, Brier score) of various models, with results saved to Neptune
- `training_hyperparameters.py`: Tunes hyperparameters using Optuna, with results logged to Neptune
- `training_results.py`: Assesses training performance (discrimination, calibration, Brier score) using 10-fold stratified cross-validation repeated 5 times
- `utils_evaluation.py`: Utility script for model performance evaluation, including calibration, cross-validation for training, bootstrap for testing, and SHAP analysis
- `utils_neptune.py`: Utility script for logging experiments to Neptune, including performance metrics and plots
- `utils_plots.py`: Utility script for generating plots

3. Tables
- `Baseline_tables.py`: Generates baseline tables for complete and imputed data, compares training and testing data, and contrasts patients with complete versus missing EQ-5D-3L questionnaires

### Manuscript & Article Supplements (to be added, currently in manuscript submission process)

- Published manuscript (link: `.....`)
- Supplements

## Prerequisites

- Python 3.x
- R
- Jupyter Notebook or Jupyter Lab

## Usage

We refer to each file for detailed instructions on usage. We refer to the manuscript for details on methodology and references.

## Citation

Please cite the following article when using or referencing this work in your research (to be added):

`[Author(s)] (Year). [Article Title]. [Journal Name]. [Link to article]`

## Acknowledgements

This work was supported by the AO Spine Discovery & Innovation Award (Award Code: AOS-DIA-22-012-TUM). We thank all contributors and participants in this study for their invaluable input.
