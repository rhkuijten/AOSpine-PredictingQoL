
from matplotlib import pyplot as plt
import pandas as pd
import neptune
from utils_evaluation import bootstrap_model_performance, compute_shap_values, cross_val_model_performance
from utils_plots import plot_calibration_curve, plot_decision_curve_analysis, plot_feature_importances, plot_roc_curve, plot_shap_bar, plot_shap_bee

def log_metrics_and_plots(run, model_type, model, cv=None, X=None, y=None, optuna = False, log_detailed_metrics = False, testing = False, X_test=None, y_test=None, n_bootstraps = 2000):
    """
    Logs model performance metrics and generates relevant plots during cross-validation or testing, with optional SHAP value analysis.

    Parameters:
    -----------
    run : neptune.run.Run
        The Neptune run object used to log metrics, parameters, and plots.

    model_type : str
        The type of model being evaluated (e.g., 'rf' for Random Forest, 'sgb' for Stochastic Gradient Boosting, 'plr' for Penalized Logistic Regression).

    model : object
        The machine learning model to be evaluated. The model should have a `predict_proba` method for probability predictions.

    cv : int or cross-validation generator, optional
        The cross-validation strategy, such as the number of folds (e.g., `cv=10`) or a cross-validation generator like `StratifiedKFold`. Used only during cross-validation.

    X : array-like or DataFrame, optional
        The feature matrix used for training or cross-validation. Each row corresponds to an instance, and each column corresponds to a feature.

    y : array-like, optional
        The true labels for the instances in `X`. Used only during cross-validation.

    optuna : bool, optional (default=False)
        If True, the function will return only the Brier score during Optuna optimization, without logging detailed metrics or generating plots.

    log_detailed_metrics : bool, optional (default=False)
        If True, detailed metrics including AUC and Brier scores will be logged even during Optuna optimization.

    testing : bool, optional (default=False)
        If True, the function evaluates the model using the test set and bootstrapping, rather than cross-validation.

    X_test : array-like or DataFrame, optional
        The feature matrix used for testing the model. Each row corresponds to an instance, and each column corresponds to a feature. Used only during testing.

    y_test : array-like, optional
        The true labels for the instances in `X_test`. Used only during testing.

    n_bootstraps : int, optional (default=2000)
        The number of bootstrap samples to generate during testing.
        
    Returns:
    --------
    float, optional
        During Optuna optimization, returns the Brier score if `optuna` is True. Otherwise, returns None.

    Notes:
    ------
    - The function logs performance metrics such as AUC, Brier score, calibration intercept, and calibration slope.
    - During cross-validation, the metrics are computed using the `cross_val_model_performance` function.
    - During testing, the metrics are computed using the `bootstrap_model_performance` function.
    - If SHAP analysis is enabled, SHAP values are computed and relevant plots are logged.
    - Various plots, including calibration curves, decision curves, ROC curves, and feature importance plots, are generated and logged.
    - Model parameters are also logged, with non-standard types converted to strings to prevent logging errors.
    """
    
    if not testing:
        # Compute metrics
        auc, cal_int, cal_slope, brier, calibration_data, y_pred_prob = cross_val_model_performance(model, X, y, cv = cv)
        
        if log_detailed_metrics or not optuna:
            run["metrics/cv_auc"] = auc[0]
            run["metrics/cv_brier"] = brier[0]
            
            # Set None value to str(None) to prevent logging error
            params = model.named_steps['classifier'].get_params()
            clean_params = {k: (str(v) if v is None or not isinstance(v, (str, int, float, bool)) else v) for k, v in params.items()}
            clean_params = {k: v for k, v in clean_params.items() if k != 'random_state'}
            run["model_params"] = clean_params
            
            pd.DataFrame({'model_name': [model_type],
                        'auc_score': [f"{auc[0]} ({auc[1]} ; {auc[2]})"],
                        'calibration_intercept': [f"{cal_int[0]} ({cal_int[1]} ; {cal_int[2]})"],
                        'calibration_slope': [f"{cal_slope[0]} ({cal_slope[1]} ; {cal_slope[2]})"],
                        'brier_score': [f"{brier[0]} ({brier[1]} ; {brier[2]})"]}).to_csv("summary_metrics.csv", index=False)
            
            run["metrics/summary"].upload("summary_metrics.csv")
        
            # Generate, log, and close plots
            for plot_func, plot_name in [
                (plot_calibration_curve, "plots/Calibration"),
                (plot_decision_curve_analysis, "plots/Decision Curve Analysis"),
                (plot_roc_curve, "plots/ROC Curve")]:
                
                fig = plot_func(y, y_pred_prob, model_type)
                run[plot_name].upload(neptune.types.File.as_image(fig))
                plt.close(fig)
                
            if model_type in ['rf', 'sgb', 'plr']:
                # Create and fit the FeatureImportances visualizer
                fig = plot_feature_importances(model, X, y)
                run["plots/Feature Importances"].upload(neptune.types.File.as_image(fig))
                plt.close(fig)
                
        if optuna:
            return brier[0]

    if testing:
        # Compute metrics
        auc, cal_int, cal_slope, brier, y_pred_prob = bootstrap_model_performance(model, X_test, y_test, n_bootstraps = n_bootstraps) 
        
        # Set None value to str(None) to prevent logging error
        params = model.named_steps['classifier'].get_params()
        clean_params = {k: (str(v) if v is None or not isinstance(v, (str, int, float, bool)) else v) for k, v in params.items()}
        clean_params = {k: v for k, v in clean_params.items() if k != 'random_state'}
        run["model_params"] = clean_params

        run['metrics/auc_ci'] = f"{auc[0]} ({auc[1]} - {auc[2]})"
        run['metrics/int_ci'] = f"{cal_int[0]} ({cal_int[1]} - {cal_int[2]})"
        run['metrics/slope_ci'] = f"{cal_slope[0]} ({cal_slope[1]} - {cal_slope[2]})"
        run['metrics/brier_ci'] = f"{brier[0]} ({brier[1]} - {brier[2]})"
            
        pd.DataFrame({'model_name': [model_type],
                      'auc_score': [f"{auc[0]} ({auc[1]} - {auc[2]})"],
                      'calibration_intercept': [f"{cal_int[0]} ({cal_int[1]} - {cal_int[2]})"],
                      'calibration_slope': [f"{cal_slope[0]} ({cal_slope[1]} - {cal_slope[2]})"],
                      'brier_score': [f"{brier[0]} ({brier[1]} - {brier[2]})"]}).to_csv("summary_metrics.csv", index=False)
            
        run["metrics/summary"].upload("summary_metrics.csv")
        
        # Calculate shap values
        shap_values = compute_shap_values(model, X_test)
        
        # Generate, log, and close plots
        for plot_func, plot_name in [
            (plot_calibration_curve, "plots/Calibration"),
            (plot_decision_curve_analysis, "plots/Decision Curve Analysis"),
            (plot_roc_curve, "plots/ROC Curve")]:
                
            fig = plot_func(y_test, y_pred_prob, model_type)
            run[plot_name].upload(neptune.types.File.as_image(fig))
            plt.close(fig)

        # Generate, log, and close shap plots
        fig = plot_shap_bar(shap_values)
        run["plots/SHAP bar"].upload(neptune.types.File.as_image(fig))
        plt.close(fig)

        # Generate, log, and close shap plots
        fig = plot_shap_bee(shap_values)
        run["plots/SHAP bee"].upload(neptune.types.File.as_image(fig))
        plt.close(fig)
        
        if model_type in ['rf', 'sgb', 'plr']:
                # Create and fit the FeatureImportances visualizer
                fig = plot_feature_importances(model, X_test, y_test)
                run["plots/Feature Importances"].upload(neptune.types.File.as_image(fig))
                plt.close(fig)
        