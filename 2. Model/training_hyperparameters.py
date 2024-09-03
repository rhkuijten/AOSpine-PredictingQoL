# Basic packages
import pandas as pd
import pickle

# Transformation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer

# Model tracking
import neptune
import neptune.integrations.optuna as optuna_utils

# Model training
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Optimization
import optuna

# Model evaluation
from utils_neptune import log_metrics_and_plots


# Load the existing variables
with open(r"C:\Users\rhkui\OneDrive\Github Repositories\AOSpine-PredictingQoL\my_variables.pkl", "rb") as pickle_file:
    data = pickle.load(pickle_file)

# Load the processed datasets
with open(r"C:\Users\rhkui\OneDrive\Github Repositories\AOSpine-PredictingQoL\processed_data.pickle", "rb") as pickle_file:
    processed_data = pickle.load(pickle_file)


features = ['B_Index', 'Katagiri_Group', 'Brain', 'Opioid', 'grouped_KPS']
features_num = ['B_Index']

X_train = processed_data["X_train"][features]
y_train = processed_data["y_train"]

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

####################### SET RUN PARAMETERS ######################
run_name = "hp_search_5_KPS"
cv_method = RepeatedStratifiedKFold(n_repeats=1, n_splits=10, random_state=22)
##################################################################
##################################################################

# Define models
model_types = ["rf", "sgb", "svm", "plr", "nn"]


def create_model(trial, model_type, features):
    if model_type == "rf":
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        criterion = trial.suggest_categorical(
            "criterion", ["gini", "entropy", "log_loss"]
        )
        max_depth = trial.suggest_int("max_depth", 3, len(features))
        min_samples_split = trial.suggest_int("min_samples_split", 2, len(features))
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, len(features))
        max_features = trial.suggest_categorical("max_features", ["log2", "sqrt", None])
        class_weight = trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample", None]
        )
        model = Pipeline(
            [
                ("preprocessor", power_and_scale_transformer),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        criterion=criterion,
                        max_depth=max_depth,
                        max_features=max_features,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        class_weight=class_weight,
                        n_jobs=-1,
                        random_state=22,
                    ),
                ),
            ]
        )

    elif model_type == "sgb":
        loss = trial.suggest_categorical("loss", ["log_loss", "exponential"])
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1)
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        subsample = trial.suggest_float("subsample", 0.5, 0.99)
        criterion = trial.suggest_categorical(
            "criterion", ["friedman_mse", "squared_error"]
        )
        min_samples_split = trial.suggest_int("min_samples_split", 2, len(features))
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, len(features))
        max_depth = trial.suggest_int("max_depth", 3, len(features))
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        model = Pipeline(
            [
                ("preprocessor", power_and_scale_transformer),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        loss=loss,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        subsample=subsample,
                        criterion=criterion,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_depth=max_depth,
                        max_features=max_features,
                        random_state=22,
                    ),
                ),
            ]
        )

    elif model_type == "svm":
        C = trial.suggest_float("C", 0.01, 10.0)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
        model = Pipeline(
            [
                ("preprocessor", power_and_scale_transformer),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=C,
                        gamma=gamma,
                        class_weight=class_weight,
                        probability=True,
                        random_state=22,
                    ),
                ),
            ]
        )

    elif model_type == "plr":
        C = trial.suggest_float("C", 0.01, 10.0)
        l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
        model = Pipeline(
            [
                ("preprocessor", power_and_scale_transformer),
                (
                    "classifier",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=l1_ratio,
                        C=C,
                        class_weight=class_weight,
                        max_iter=10000,
                        n_jobs=-1,
                        random_state=22,
                    ),
                ),
            ]
        )

    elif model_type == "nn":
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        hidden_layer_sizes_map = {
            "50": (50,),
            "100": (100,),
            "50_50": (50, 50),
            "100_100": (100, 100),
        }

        hidden_layer_sizes_str = trial.suggest_categorical(
            "hidden_layer_sizes", ["50", "100", "50_50", "100_100"]
        )
        hidden_layer_sizes = hidden_layer_sizes_map[hidden_layer_sizes_str]
        activation = trial.suggest_categorical(
            "activation", ["identity", "logistic", "tanh", "relu"]
        )
        learning_rate_init = trial.suggest_float(
            "learning_rate_init", 1e-5, 1e-1, log=True
        )
        solver = trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"])
        learning_rate = "constant"
        if solver == "sgd":
            learning_rate = trial.suggest_categorical(
                "learning_rate", ["constant", "invscaling", "adaptive"]
            )
        momentum = 0.9
        if solver == "sgd":
            momentum = trial.suggest_float("momentum", 0.5, 0.9)
        model = Pipeline(
            [
                ("preprocessor", power_and_scale_transformer),
                (
                    "classifier",
                    MLPClassifier(
                        alpha=alpha,
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,
                        learning_rate_init=learning_rate_init,
                        solver=solver,
                        learning_rate=learning_rate,
                        momentum=momentum,
                        max_iter=10000,
                        random_state=22,
                    ),
                ),
            ]
        )

    if trial.should_prune():
        raise optuna.TrialPruned()

    return model


# Start: loop and runs
for model_type in model_types:
    # Initialize a new run for each model type
    run = neptune.init_run(
        project="PredictingQoL/PredQoL",
        custom_run_id=f"{model_type}_{run_name}",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhM2VhYTlmZS00YjMwLTRiNWUtOTA5NS00Mzc0MGU0MTQyZDAifQ==",
    )
    neptune_callback = optuna_utils.NeptuneCallback(run)

    def objective(trial):
        model = create_model(trial, model_type, features)
        brier = log_metrics_and_plots(run, model_type, model, X = X_train, y = y_train, cv=cv_method, optuna=True, log_detailed_metrics=False)
        
        # Initially assume this trial should log detailed metrics
        should_log_detailed_metrics = True
        
        try:
            # Only log detailed metrics for the best trial after the first trial
            should_log_detailed_metrics = study.best_trial is None or brier > study.best_trial.value
        except ValueError:
            # ValueError is caught for the first trial where there is no best_trial yet
            pass
        
        # Log detailed metrics if this is the first trial or if it's currently the best trial
        if should_log_detailed_metrics:
            log_metrics_and_plots(run, model_type, model, X = X_train, y = y_train, cv=cv_method, optuna=False, log_detailed_metrics=True)
        
        return brier

    study = optuna.create_study(
        direction="minimize",
        study_name=f"PredictingQoL-{model_type}",
        storage="sqlite:///PredictingQoL.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100, callbacks=[neptune_callback])

    # Stop the Neptune run
    run.stop()
