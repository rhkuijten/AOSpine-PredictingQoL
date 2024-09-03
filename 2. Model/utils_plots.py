import matplotlib
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, roc_curve
from yellowbrick.model_selection import FeatureImportances

model_dict = {
    "rf": "Random forest",
    "sgb": "Stochastic gradient boosting",
    "nn": "Neural network",
    "svm": "Support vector machine",
    "plr": "Penalized logistic regression",
}

def plot_decision_curve_analysis(y_true, y_pred_prob, model_name):
    """
    Generates a Decision Curve Analysis (DCA) plot for a given model, comparing the net benefit of using the model's predictions
    at various threshold probabilities to the strategies of treating all or none of the cases.

    Parameters:
    -----------
    y_true : array-like
        The true binary outcomes, where each element is 0 or 1.

    y_pred_prob : array-like
        The predicted probabilities for the positive class (1) from the model.

    model_name : str
        The name of the model being evaluated. This is used to label the curve in the plot.

    Returns:
    --------
    matplotlib.figure.Figure
        A matplotlib figure object containing the Decision Curve Analysis plot.

    Notes:
    ------
    - The function computes the standardized net benefit for the model across a range of threshold probabilities.
    - The standardized net benefit for treating all cases is also computed and plotted for comparison.
    - The plot includes three lines: the model's standardized net benefit, the standardized net benefit for treating all, and a baseline for treating none.
    - The resulting plot shows the net benefit of the model's predictions relative to these strategies, helping to assess the clinical utility of the model.
    - The plot is customized with axis labels, a title, grid lines, and a legend.
    """
    fig, ax = plt.subplots()

    model = model_dict[model_name]
    thresh_group = np.arange(0, 1, 0.05)

    std_nb_model = np.array([])
    std_nb_all_model = np.array([])

    mcid_incidence = np.sum(y_true == 1)
    n = len(y_true)

    for thresh in thresh_group:
        # Get the predicted classes based on the threshold
        y_pred = y_pred_prob >= thresh
        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Net benefit
        nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        if tp != 0 and not np.isnan(nb) and not np.isnan(tp) and not np.isnan(n):
            std_nb = nb / (mcid_incidence / n)
        else:
            std_nb = 0
        std_nb_model = np.append(std_nb_model, std_nb)

        # Net benefit treat all
        _, _, _, tp = confusion_matrix(y_true, [1] * len(y_true)).ravel()
        nb_all = (tp / n) - (1 - (tp / n)) * (thresh / (1 - thresh))
        if tp != 0 and not np.isnan(nb) and not np.isnan(tp) and not np.isnan(n):
            std_nb_all = nb_all / (mcid_incidence / n)
        else:
            # Handle the invalid case, perhaps set std_nb to NaN or some default value
            std_nb_all = 0
        std_nb_all_model = np.append(std_nb_all_model, std_nb_all)

    # Plot curve
    ax.plot(thresh_group, std_nb_model, color="blue", label=f"{model}")
    ax.plot(thresh_group, std_nb_all_model, color="grey", label="Treat all")
    ax.plot((0, 1), (0, 0), color="grey", linestyle=":", label="Treat none")

    ax.set_ylim(-0.15, 1.15)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel(xlabel="Threshold Probability", size=12)
    ax.set_ylabel(ylabel="Standardized Net Benefit", size=12)
    ax.set_title(label="Decision Curve Analysis", pad=25, size=15, fontweight="bold")
    ax.grid("major")
    ax.spines["right"].set_color((0.8, 0.8, 0.8))
    ax.spines["top"].set_color((0.8, 0.8, 0.8))
    ax.legend(loc="upper right")

    return fig


def tricube(d):
    """
    Computes the tricube weighting function for a given distance or set of distances.

    Parameters:
    -----------
    d : float or array-like
        The input distance(s) for which the tricube weights are to be computed. The distance should typically be within the range [-1, 1].

    Returns:
    --------
    float or array-like
        The tricube weights corresponding to the input distance(s). The weights are calculated as `(1 - |d|^3)^3` and are clipped to the range [0, 1].

    Notes:
    ------
    - The tricube function is commonly used in local regression (such as LOESS) as a smooth, non-negative weighting function.
    - The function is defined as `(1 - |d|^3)^3`, where `d` is the input distance.
    - If `d` is outside the range [-1, 1], the resulting weight is clipped to 0.
    """
    # Define the weigthing function
    return np.clip((1 - np.abs(d) ** 3) ** 3, 0, 1)


def lowess(x, y, f):
    """
    Performs Locally Weighted Scatterplot Smoothing (LOWESS) to fit a smooth curve to the input data, 
    using a tricube weighting function.

    Parameters:
    -----------
    x : array-like
        The independent variable (predictor) values.
        
    y : array-like
        The dependent variable (response) values.

    f : float
        The smoothing parameter, controlling the fraction of the data used when estimating each y-value. 
        It should be between 0 and 1, where a smaller value corresponds to less smoothing.

    Returns:
    --------
    tuple (array, array)
        A tuple containing:
        - y_sm : array
            The smoothed y-values corresponding to the input x-values.
        - y_stderr : array
            The standard errors of the smoothed y-values, indicating the uncertainty of the fit at each point.

    Notes:
    ------
    - The function first sorts the input data by the independent variable `x` to ensure the smooth curve is computed correctly.
    - For each observation, the function calculates a local linear regression weighted by the tricube function, which gives more weight to points near the target point.
    - The width of the neighborhood is determined by the smoothing parameter `f`.
    - The standard error of the smoothed estimate is also computed, which can be used to assess the variability of the fitted values.
    - The function returns both the smoothed y-values and their associated standard errors.
    """
    # Get some paras
    xwidth = f * (x.max() - x.min())  # effective width after reduction factor
    N = len(x)  # number of obs
    # Don't assume the data is sorted
    order = np.argsort(x)
    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)

    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i] - x[order])) / xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order] * w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the syste
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        yest = A[i].dot(sol)  # equiv of A.dot(yest) just for k
        place = order[i]
        y_sm[place] = yest
        sigma2 = np.sum((A.dot(sol) - y[order]) ** 2) / N
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 * A[i].dot(np.linalg.inv(ATA)).dot(A[i]))
    return y_sm, y_stderr


def plot_calibration_curve(y_actual, y_pred_prob, model_name):
    """
    Generates a calibration curve plot to assess the agreement between predicted probabilities and observed outcomes.

    Parameters:
    -----------
    y_actual : array-like
        The true binary outcomes, where each element is 0 or 1.

    y_pred_prob : array-like
        The predicted probabilities for the positive class (1) from the model.

    model_name : str
        The name of the model being evaluated. This is used to label the curve in the plot.

    Returns:
    --------
    matplotlib.figure.Figure
        A matplotlib figure object containing the calibration curve plot.

    Notes:
    ------
    - The calibration curve compares the predicted probability of the positive class to the observed proportion of positive cases, typically using binning.
    - The function applies a LOWESS (Locally Weighted Scatterplot Smoothing) algorithm to smooth the observed proportion of positives.
    - The plot includes:
        - A black line representing the smoothed calibration curve of the model.
        - A red diagonal line representing perfect calibration (where predicted probabilities exactly match observed outcomes).
        - Histograms of the predicted probabilities for positive and negative cases, displayed as green and red bars, respectively.
        - A shaded region around the smoothed curve representing the standard error, giving an indication of the uncertainty around the calibration estimate.
    - The x-axis represents the predicted probabilities, and the y-axis represents the observed proportions.
    """
    fig, ax = plt.subplots()
    model = model_dict[model_name]

    # Slope & intercept
    y, x = calibration_curve(y_actual, y_pred_prob, n_bins=10)

    # run it
    y_sm, y_std = lowess(x, y, f=0.75)
    order = np.argsort(x)

    # Plot
    line, = ax.plot(x[order], y_sm[order], color="black", label=f"{model}", linewidth=0.5)
    perfect_line, = ax.plot([0, 1], [0, 1], color="red", label="Perfect calibration", linestyle="-", linewidth=0.5)

    y_pred_prob_positive = y_pred_prob[y_actual == 1]
    y_pred_prob_negative = y_pred_prob[y_actual == 0]
    weights_positive = np.ones_like(y_pred_prob_positive) / len(y_pred_prob)
    ax.hist(
        y_pred_prob_positive,
        weights=weights_positive,
        bins=50,
        alpha=0.4,
        color="green",
        label = "Positive predictions"
    )
    weights_negative = -np.ones_like(y_pred_prob_negative) / len(y_pred_prob)
    ax.hist(
        y_pred_prob_negative,
        weights=weights_negative,
        bins=50,
        alpha=0.4,
        color="red",
        label = "Negative predictions"
    )
    ax.fill_between(
        x[order],
        y_sm[order] - y_std[order],
        y_sm[order] + y_std[order],
        alpha=0.3,
        color="grey",
    )
    
    ax.legend(loc = 'upper left')
    ax.set_xlabel("Predicted probability", size=12)
    ax.set_ylabel("Observed proportion", size=12)
    ax.set_title(label="Calibration", pad=25, size=15, fontweight="bold")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1])

    return fig


def plot_roc_curve(y_actual, y_pred_prob, model_name):
    """
    Generates a Receiver Operating Characteristic (ROC) curve to evaluate the classification performance of a model.

    Parameters:
    -----------
    y_actual : array-like
        The true binary outcomes, where each element is 0 or 1.

    y_pred_prob : array-like
        The predicted probabilities for the positive class (1) from the model.

    model_name : str
        The name of the model being evaluated. This is used to label the curve in the plot.

    Returns:
    --------
    matplotlib.figure.Figure
        A matplotlib figure object containing the ROC curve plot.

    Notes:
    ------
    - The ROC curve is a graphical representation of a model's ability to discriminate between the positive and negative classes.
    - The function calculates the False Positive Rate (FPR) and True Positive Rate (TPR) at various threshold levels and plots these points to form the ROC curve.
    - The Area Under the Curve (AUC) is also calculated and displayed in the plot legend, providing a summary measure of the model's performance.
    - The x-axis represents the False Positive Rate, and the y-axis represents the True Positive Rate.
    - The plot includes axis labels, a grid, and a legend that indicates the model's name and AUC score.
    """
    fig, ax = plt.subplots()
    model = model_dict[model_name]

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_actual, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, label=f"{model} (AUC = {roc_auc:.2f})", color="red")
    ax.grid(visible=True, which="major")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", size=12)
    ax.set_ylabel("True Positive Rate", size=12)
    ax.set_title("Discrimination", pad=25, size=15, fontweight="bold")
    ax.legend(loc="best")

    return fig


def plot_shap_bar(shap_values):
    """
    Generates a SHAP bar plot to display the mean feature importances based on SHAP values.

    Parameters:
    -----------
    shap_values : shap.Explanation
        A SHAP Explanation object containing the SHAP values for a model's predictions, along with feature names and other metadata.

    Returns:
    --------
    matplotlib.figure.Figure
        A matplotlib figure object containing the SHAP bar plot.

    Notes:
    ------
    - The SHAP bar plot visualizes the mean absolute SHAP values for each feature, indicating the overall importance of each feature in the model's predictions.
    - The plot helps to identify which features have the most significant impact on the model's output.
    - The plot title is set to "SHAP Mean Feature Importances" to clearly indicate the content of the plot.
    - The `show=False` parameter is passed to `shap.plots.bar` to ensure the plot is rendered within the provided axes and returned as part of the figure.
    """
    # Create a 1 row, 2 column grid of subplots
    fig, ax = plt.subplots()

    # SHAP bar plot on the left
    shap.plots.bar(shap_values, ax=ax, show=False)
    ax.set_title("SHAP Mean Feature Importances")

    return fig


def plot_shap_bee(shap_values):
    """
    Generates a SHAP beeswarm plot to visualize the distribution of SHAP values for each feature across all samples.

    Parameters:
    -----------
    shap_values : shap.Explanation
        A SHAP Explanation object containing the SHAP values for a model's predictions, along with feature names and other metadata.

    Returns:
    --------
    matplotlib.figure.Figure
        A matplotlib figure object containing the SHAP beeswarm plot.

    Notes:
    ------
    - The SHAP beeswarm plot displays the SHAP values for each feature, showing both the magnitude and the direction of impact on the model's predictions.
    - Each point on the beeswarm plot represents a single SHAP value for a feature in one sample, with color indicating the feature value.
    - The `show=False` parameter is passed to `shap.plots.beeswarm` to ensure the plot is rendered within the provided figure context and returned as part of the figure.
    - The function returns the figure object containing the SHAP beeswarm plot, which can be saved or displayed.
    """
    # Create a 1 row, 2 column grid of subplots
    fig, _ = plt.subplots()

    # SHAP beeswarm plot on the right
    shap.plots.beeswarm(shap_values, show=False)

    fig = plt.gcf()

    return fig


def plot_shap_waterfall(shap_values):
    """
    Generates a SHAP waterfall plot to visualize the contribution of each feature for a single prediction.

    Parameters:
    -----------
    shap_values : shap.Explanation
        A SHAP Explanation object containing the SHAP values for a model's predictions, along with feature names and other metadata.
        This function uses the SHAP values for a single instance (e.g., `shap_values[0]`).

    Returns:
    --------
    matplotlib.figure.Figure
        A matplotlib figure object containing the SHAP waterfall plot.

    Notes:
    ------
    - The SHAP waterfall plot visualizes how each feature contributes to moving the model's output from the base value (expected value) to the predicted value for a single instance.
    - Features are displayed in order of their impact, with the most influential features at the top.
    - The color of each feature's bar typically indicates the feature value (e.g., red for high values and blue for low values), providing additional insight into how feature values affect the prediction.
    - The `show=False` parameter is passed to `shap.plots.waterfall` to ensure the plot is rendered within the provided figure context and returned as part of the figure.
    - By default, this function generates the waterfall plot for the first instance in `shap_values`. To visualize a different instance, modify the index accordingly (e.g., `shap_values[1]` for the second instance).
    - The waterfall plot is useful for understanding individual predictions and identifying which features are driving the model's decision for that specific instance.
    """
    fig, _ = plt.subplots()

    shap.plots.waterfall(shap_values[0], show=False)

    return fig


def plot_feature_importances(model, X, y):
    """
    Generates a plot of feature importances for a given model, using the `FeatureImportances` visualizer from the `yellowbrick` library.

    Parameters:
    -----------
    model : object
        A machine learning pipeline or model that contains a classifier step with a `feature_importances_` or `coef_` attribute.
        
    X : array-like or DataFrame
        The feature matrix used for training the model. Each row corresponds to an instance, and each column corresponds to a feature.

    y : array-like
        The true labels for the instances in `X`.

    Returns:
    --------
    matplotlib.figure.Figure
        A matplotlib figure object containing the feature importances plot.

    Notes:
    ------
    - The function uses the `FeatureImportances` visualizer from the `yellowbrick` library to generate the plot.
    - The `model` should be a pipeline or a model that has been fitted and includes a `classifier` step with accessible feature importances.
    - The feature importances are plotted to show the relative importance of each feature in the model's predictions.
    - The plot is returned as a figure object, which can be saved or displayed.
    """
    fig, ax = plt.subplots()

    viz = FeatureImportances(model.named_steps["classifier"], ax=ax)
    viz.fit(X.copy(), y.copy())
    viz.finalize()
    
    return fig
