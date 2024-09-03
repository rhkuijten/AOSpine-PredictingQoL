import numpy as np
import shap
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.utils import resample


def logit(p):
    """
    Computes the logit (log-odds) of a given probability or set of probabilities.

    Parameters:
    -----------
    p : float or array-like
        A probability or array of probabilities for which to compute the logit. Each value should be between 0 and 1.

    Returns:
    --------
    float or array-like
        The logit (log-odds) corresponding to the input probability or probabilities.

    Notes:
    ------
    - The input probabilities are clipped to a minimum of 1e-15 and a maximum of 1 - 1e-15 to avoid 
      issues with calculating the logit at the extremes (0 and 1), where the log function would be undefined.
    - The logit function is defined as log(p / (1 - p)), where p is the probability.
    """
    clipped_p = np.clip(p, 1e-15, 1 - 1e-15)
    
    return np.log(clipped_p / (1 - clipped_p))


def calibration_intercept_slope(y_true, y_pred_proba):
    """
    Calculates both the calibration intercept and slope for a set of predicted probabilities against the true binary outcomes.

    Parameters:
    -----------
    y_true : array-like
        The true binary outcomes, where each element is 0 or 1.

    y_pred_proba : array-like
        The predicted probabilities for the positive class (1). Each element should be a probability value 
        between 0 and 1.

    Returns:
    --------
    tuple (float, float)
        A tuple containing:
        - intercept: The calibration intercept, indicating the overall bias in the predicted probabilities.
        - slope: The calibration slope, measuring the relationship between the predicted probabilities and the actual outcomes.
          A slope of 1 indicates perfect calibration.

    Notes:
    ------
    - The function first transforms the predicted probabilities into log-odds using the logit function.
    - A logistic regression model is then fitted to the log-odds against the true binary outcomes to determine the 
      calibration intercept and slope.
    - The logistic regression model is fitted without regularization (no penalty).
    """
    
    # Fit logistic regression to predicted probabilities
    log_odds = logit(y_pred_proba)

    lr = LogisticRegression(penalty=None)
    lr.fit(log_odds.reshape(-1, 1), y_true)

    # Intercept and slope
    intercept = lr.intercept_[0]
    slope = lr.coef_[0][0]

    return intercept, slope


def calibration_slope(y_true, y_pred_proba):
    """
    Calculates the calibration slope for a set of predicted probabilities against the true binary outcomes.

    Parameters:
    -----------
    y_true : array-like
        The true binary outcomes, where each element is 0 or 1.
    
    y_pred_proba : array-like
        The predicted probabilities for the positive class (1). Each element should be a probability value 
        between 0 and 1.

    Returns:
    --------
    float
        The calibration slope, which measures the relationship between the predicted probabilities and the 
        actual outcomes. A slope of 1 indicates perfect calibration.
    """
    _, slope = calibration_intercept_slope(y_true, y_pred_proba)
    return slope


def calibration_intercept(y_true, y_pred_proba):
    """
    Calculates the calibration intercept for a set of predicted probabilities against the true binary outcomes.

    Parameters:
    -----------
    y_true : array-like
        The true binary outcomes, where each element is 0 or 1.
    
    y_pred_proba : array-like
        The predicted probabilities for the positive class (1). Each element should be a probability value 
        between 0 and 1.

    Returns:
    --------
    float
        The calibration intercept, which measures the overall tendency of the predicted probabilities to be 
        higher or lower than the true outcomes. An intercept of 0 indicates perfect calibration.
    """
    intercept, _ = calibration_intercept_slope(y_true, y_pred_proba)
    return intercept


scoring = {
    "auc": "roc_auc",
    "cal_int": make_scorer(calibration_intercept, needs_proba=True),
    "cal_slope": make_scorer(calibration_slope, needs_proba=True),
    "brier": make_scorer(brier_score_loss, needs_proba=True),
}


def calculate_metric_statistics(scores):
    """
    Calculates statistical metrics including the mean and confidence intervals for a given set of scores.

    Parameters:
    -----------
    scores : array-like
        A list or array of numeric scores for which to compute the statistics.

    Returns:
    --------
    tuple (float, float, float)
        A tuple containing:
        - mean_score: The mean of the sorted scores, rounded to two decimal places.
        - lower_ci: The lower bound of the 95% confidence interval (2.5th percentile), rounded to two decimal places.
        - upper_ci: The upper bound of the 95% confidence interval (97.5th percentile), rounded to two decimal places.

    Notes:
    ------
    - The function sorts the input scores before calculating the mean and confidence intervals.
    - The confidence intervals are computed using the 2.5th and 97.5th percentiles of the sorted scores.
    """
    LOWER_PERCENTILE = 2.5
    UPPER_PERCENTILE = 97.5

    sorted_scores = np.sort(scores)
    mean_score = round(np.mean(sorted_scores), 2)
    lower_ci = round(np.percentile(sorted_scores, LOWER_PERCENTILE), 2)
    upper_ci = round(np.percentile(sorted_scores, UPPER_PERCENTILE), 2)
    return (mean_score, lower_ci, upper_ci)


def cross_val_model_performance(model, X, y, cv):
    """
    Evaluates model performance using cross-validation and computes various metrics, including AUC, calibration intercept, 
    calibration slope, and Brier score.

    Parameters:
    -----------
    model : object
        The machine learning model to be evaluated. The model should have a `predict_proba` method for probability predictions.

    X : array-like or DataFrame
        The feature matrix used for training the model. Each row corresponds to an instance, and each column corresponds to a feature.

    y : array-like
        The true labels for the instances in `X`.

    cv : int or cross-validation generator
        The cross-validation strategy, such as the number of folds (e.g., `cv=10`) or a cross-validation generator like `StratifiedKFold`.

    Returns:
    --------
    tuple
        A tuple containing:
        - auc_sum : tuple (float, float, float)
            The mean, lower, and upper bounds of the 95% confidence interval for the AUC scores.
        - int_sum : tuple (float, float, float)
            The mean, lower, and upper bounds of the 95% confidence interval for the calibration intercepts.
        - slope_sum : tuple (float, float, float)
            The mean, lower, and upper bounds of the 95% confidence interval for the calibration slopes.
        - brier_sum : tuple (float, float, float)
            The mean, lower, and upper bounds of the 95% confidence interval for the Brier scores.
        - calibration_plot_data : tuple (array, array)
            A tuple containing the fraction of positives and mean predicted value for calibration plot purposes.
        - y_probas : array
            The predicted probabilities for the positive class (1) across all folds.

    Notes:
    ------
    - The function uses cross-validation to obtain predicted probabilities and evaluate the model's performance.
    - Evaluation metrics include AUC, calibration intercept, calibration slope, and Brier score.
    - The calibration plot data can be used to visualize the calibration of the model's predicted probabilities.
    """
    # Collect the predicted probabilities and true labels for each fold
    y_probas = cross_val_predict(
        model,
        X,
        y,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=22),
        method="predict_proba",
    )[:, 1]
    y_true = y

    # Calculate evaluation metrics
    cv_results = cross_validate(model, X, y, scoring=scoring, cv=cv)

    auc_sum = calculate_metric_statistics(cv_results["test_auc"])
    int_sum = calculate_metric_statistics(cv_results["test_cal_int"])
    slope_sum = calculate_metric_statistics(cv_results["test_cal_slope"])
    brier_sum = calculate_metric_statistics(cv_results["test_brier"])

    # Create calibration plot data
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_probas, n_bins=10
    )

    return (
        auc_sum,
        int_sum,
        slope_sum,
        brier_sum,
        (fraction_of_positives, mean_predicted_value),
        y_probas,
    )


def bootstrap_model_performance(model, X_test, y_test, n_bootstraps=2000):
    """
    Evaluates the performance of a model using bootstrapping to estimate the variability of key metrics, 
    including AUC, calibration intercept, calibration slope, and Brier score.

    Parameters:
    -----------
    model : object
        The machine learning model to be evaluated. The model should have a `predict_proba` method for probability predictions.

    X_test : array-like or DataFrame
        The feature matrix used for testing the model. Each row corresponds to an instance, and each column corresponds to a feature.

    y_test : array-like
        The true labels for the instances in `X_test`.

    n_bootstraps : int, optional (default=2000)
        The number of bootstrap samples to generate. Each bootstrap sample is created by resampling the test set with replacement.

    Returns:
    --------
    tuple
        A tuple containing:
        - auc_stat : tuple (float, float, float)
            The mean, lower, and upper bounds of the 95% confidence interval for the AUC scores across bootstrap samples.
        - intercept_stat : tuple (float, float, float)
            The mean, lower, and upper bounds of the 95% confidence interval for the calibration intercepts across bootstrap samples.
        - slope_stat : tuple (float, float, float)
            The mean, lower, and upper bounds of the 95% confidence interval for the calibration slopes across bootstrap samples.
        - brier_stat : tuple (float, float, float)
            The mean, lower, and upper bounds of the 95% confidence interval for the Brier scores across bootstrap samples.
        - y_pred_prob : array
            The predicted probabilities for the positive class (1) on the original test set.

    Notes:
    ------
    - The function performs bootstrapping by resampling the test set with replacement to generate multiple bootstrap samples.
    - For each bootstrap sample, the function computes AUC, calibration intercept, calibration slope, and Brier score.
    - The results provide an estimate of the variability of these metrics, expressed as means and confidence intervals.
    """
    # Bootstrap parameters
    n_bootstraps = n_bootstraps

    # Set empty lists
    auc_scores = []
    calibration_intercepts = []
    calibration_slopes = []
    brier_scores = []

    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Bootstrap loop
    for i in range(n_bootstraps):
        X_boot, y_boot = resample(
            X_test,
            y_test,
            replace=True,
            stratify=y_test,
            n_samples=len(y_test),
            random_state=i,
        )

        y_pred_boot = model.predict_proba(X_boot)[:, 1]

        # Discrimination
        auc_scores.append(roc_auc_score(y_boot, y_pred_boot))

        # Calibration
        intercept, slope = calibration_intercept_slope(y_boot, y_pred_boot)
        calibration_intercepts.append(intercept)
        calibration_slopes.append(slope)

        # Brier score
        brier_scores.append(brier_score_loss(y_boot, y_pred_boot))

    # Calculate mean and confidence intervals (AUC, Calibration Intercept & Slope, Brier)
    auc_stat = calculate_metric_statistics(auc_scores)
    intercept_stat = calculate_metric_statistics(calibration_intercepts)
    slope_stat = calculate_metric_statistics(calibration_slopes)
    brier_stat = calculate_metric_statistics(brier_scores)

    return auc_stat, intercept_stat, slope_stat, brier_stat, y_pred_prob


def compute_shap_values(model, X_test):
    """
    Computes SHAP (SHapley Additive exPlanations) values for the test set to interpret the model's predictions for the positive class.

    Parameters:
    -----------
    model : object
        The machine learning model for which SHAP values will be computed. The model should have a `predict_proba` method for probability predictions.

    X_test : array-like or DataFrame
        The feature matrix used for testing the model. Each row corresponds to an instance, and each column corresponds to a feature.
        The features will be converted to floats for compatibility with SHAP.

    Returns:
    --------
    shap.Explanation
        A SHAP Explanation object containing the SHAP values for the positive class (class 1), along with the base values, 
        feature values, and feature names. This object can be used to analyze and visualize the contribution of each feature 
        to the model's predictions.

    Notes:
    ------
    - The features in `X_test` are converted to floats before computing SHAP values to ensure compatibility with the SHAP library.
    - A custom `model_predict` function is defined to extract the predicted probabilities, which are then used by the SHAP explainer.
    - The SHAP explainer is initialized using the provided model and the test set.
    - The SHAP values are computed specifically for the positive class (class 1) and returned as a SHAP Explanation object.
    """
    # Set features to floats
    X_test = X_test.astype(float)

    def model_predict(x):
        return model.predict_proba(x)

    # Initialize the SHAP explainer using the model
    explainer = shap.Explainer(model_predict, X_test)

    # Compute SHAP values for the test set
    shap_values = explainer(X_test)

    shap_values_positive_class = shap.Explanation(
        values=shap_values.values[..., 1],
        base_values=shap_values.base_values[..., 1],
        data=shap_values.data,
        feature_names=shap_values.feature_names,
    )

    return shap_values_positive_class
