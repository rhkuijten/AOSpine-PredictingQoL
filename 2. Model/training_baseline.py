#Basic packages
import pandas as pd
import pickle

# Transformation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer

# Model tracking
import neptune

# Model training
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Model evaluation
from utils_neptune import log_metrics_and_plots

# Load the processed datasets
with open("processed_data.pickle", 'rb') as pickle_file:
    processed_data = pickle.load(pickle_file)

features = ['B_Index', 'Katagiri_Group', 'Brain', 'Opioid', 'grouped_KPS']
features_num = ['B_Index']
X_train = processed_data["X_train"][features].copy()
y_train = processed_data["y_train"].copy()

class CustomPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, all_features):
        """
        Initialize the CustomPowerTransformer.

        Parameters:
        - columns: list of feature names to power-transform
        - all_features: list of all feature names
        """
        self.columns = columns
        self.all_features = all_features

    def fit(self, X, y=None):
        """
        Fit the transformer.
        Here we compute the names for columns to power-transform and fit the PowerTransformer
        and StandardScaler on the appropriate data.
        """
        # Determine indices of columns to power-transform
        self.column_names = self.all_features[: X.shape[1]]
        self.column_indices = [
            i
            for i, col_name in enumerate(self.column_names)
            if col_name in self.columns
        ]

        # Convert X to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Fit the power transformer to the specified columns
        self.power_transformer = PowerTransformer(
            method="yeo-johnson", standardize=True
        )
        self.power_transformer.fit(X[:, self.column_indices])

        # Transform the specified columns
        X_transformed = X.copy()
        X_transformed[:, self.column_indices] = self.power_transformer.transform(
            X[:, self.column_indices]
        )

        return self

    def transform(self, X, y=None):
        """
        Apply power transformation and scaling to specified columns.
        """
        # Convert X to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Apply power transformation to specified columns
        X_transformed = X.copy()
        X_transformed[:, self.column_indices] = self.power_transformer.transform(
            X[:, self.column_indices]
        )

        return X_transformed


power_and_scale_transformer = CustomPowerTransformer(
    columns=features_num, all_features=features
)

pipelines = {
    "rf": Pipeline([
        ('preprocessor', power_and_scale_transformer),
        ('classifier', RandomForestClassifier(n_jobs = -1))
    ]),
    "sgb": Pipeline([
        ('preprocessor', power_and_scale_transformer),
        ('classifier', GradientBoostingClassifier())
    ]),
    "svm": Pipeline([
        ('preprocessor', power_and_scale_transformer),
        ('classifier', SVC(kernel='rbf', probability = True))
    ]),
    "plr": Pipeline([
        ('preprocessor', power_and_scale_transformer),
        ('classifier', LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio=0.5, max_iter = 10000, n_jobs = -1))
    ]),
    "nn" : Pipeline([
        ('preprocessor', power_and_scale_transformer),
        ('classifier', MLPClassifier(max_iter = 10000))])
     }

####################### SET RUN PARAMETERS ######################
run_name = '...'
cv_method = RepeatedStratifiedKFold(n_repeats = 1, n_splits = 10, random_state = 22)
##################################################################
##################################################################

# Define models
model_types = ['rf', 'sgb', 'svm', 'plr', 'nn']

for model_type in model_types:
    
    # Initialize a new run for each model type
    run = neptune.init_run(
        project = "PredictingQoL/PredQoL",
        custom_run_id = f"{model_type}_{run_name}",
        api_token = "...")
    
    # Select the model from the pipelines dictionary
    model = pipelines[model_type]
    
    log_metrics_and_plots(run, model_type, model, X = X_train, y = y_train, cv = cv_method)
    
    # Stop the Neptune run
    run.stop()