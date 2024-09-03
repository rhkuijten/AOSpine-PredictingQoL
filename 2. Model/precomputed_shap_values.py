import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
import pickle
import shap
import itertools
from tqdm import tqdm

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
        self.column_names = self.all_features[:X.shape[1]]
        self.column_indices = [i for i, col_name in enumerate(self.column_names) if col_name in self.columns]
        
        # Convert X to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Fit the power transformer to the specified columns
        self.power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        self.power_transformer.fit(X[:, self.column_indices])
        
        # Transform the specified columns
        X_transformed = X.copy()
        X_transformed[:, self.column_indices] = self.power_transformer.transform(X[:, self.column_indices])
        
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
        X_transformed[:, self.column_indices] = self.power_transformer.transform(X[:, self.column_indices])
        
        return X_transformed

def model_predict(x):
    """
    Predicts the probability of each class for the given input data using a pre-trained model.

    Parameters:
    -----------
    x : array-like or DataFrame
        Input data for which to predict probabilities. The input should be in the format expected by the model, 
        such as a NumPy array, list of feature vectors, or a pandas DataFrame.

    Returns:
    --------
    array-like
        An array containing the predicted probabilities for each class. Each element in the returned array 
        corresponds to the predicted probability distribution over all possible classes for a single input instance.
    """
    return model.predict_proba(x)

# Function to make predictions
def make_prediction(model, input_data):
    """
    Makes a probability prediction for the given input data using a specified model.

    Parameters:
    -----------
    model : object
        The pre-trained model used to make predictions. The model should have a `predict_proba` method.
    
    input_data : dict or array-like
        The input data for which to predict probabilities. This can be a dictionary mapping feature names to values,
        or an array-like structure where each element corresponds to a feature value.

    Returns:
    --------
    array-like
        An array containing the predicted probabilities for each class. The array corresponds to the 
        probability distribution over all possible classes for the provided input data.
    """
    df = pd.DataFrame([input_data])
    prediction = model.predict_proba(df)
    return prediction

def construct_explainer(model, X_train):
    """
    Constructs a SHAP explainer for a given model using the provided training data.

    Parameters:
    -----------
    model : object
        The pre-trained model for which the SHAP explainer will be constructed. The model should be compatible 
        with SHAP and have a `predict_proba` method.

    X_train : DataFrame or array-like
        The training data used to fit the model. This data will be used to initialize the SHAP explainer.
        It should be a pandas DataFrame or an array-like structure where each row represents a training instance 
        and each column represents a feature.

    Returns:
    --------
    shap.Explainer
        A SHAP explainer object that can be used to interpret the model's predictions by generating SHAP values.

    Notes:
    ------
    The training data features are converted to floats before initializing the SHAP explainer to ensure compatibility.
    """
    # Set features to floats
    X_train = X_train.astype(float)

    # Initialize the SHAP explainer using the model
    explainer = shap.Explainer(model_predict, X_train)

    return explainer


# Load the processed datasets
with open(r"...", "rb") as pickle_file:
    processed_data = pickle.load(pickle_file)
  
with open(r"...", 'rb') as f:
    model = pickle.load(f)

features = ['B_Index', 'Katagiri_Group', 'Brain', 'Opioid', 'grouped_KPS']
features_num = ['B_Index']

X_train = processed_data["X_train"][features].astype(float)
X_test = processed_data["X_test"][features].astype(float)

# Compute SHAP explainer using the training data
explainer = construct_explainer(model, X_train)

# Feature space
features = {
    'B_Index': np.arange(-0.329, 1.001, 0.001),
    'grouped_KPS': [0, 1, 2],
    'Katagiri_Group': [1, 2, 3],
    'Opioid': [0, 1],
    'Brain' : [0, 1]
}

# Generate all possible combinations of features
all_combinations = list(itertools.product(*features.values()))

# Convert to DataFrame
all_possible_inputs = pd.DataFrame(all_combinations, columns=features.keys())
print(all_possible_inputs)

# Dictionary to store SHAP values
shap_values_dict = {}

# Compute SHAP values for each possible input
for i, input_data in tqdm(all_possible_inputs.iterrows(), total=len(all_possible_inputs)):
    shap_values = explainer(input_data.values.reshape(1, -1))
    shap_values_dict[i] = shap_values

# Create a dictionary to store both inputs and SHAP values
data_to_store = {
    'inputs': all_possible_inputs,
    'shap_values': shap_values_dict
}

# Save the dictionary to a pickle file
with open("precomputed_shap_values_and_inputs.pkl", "wb") as f:
    pickle.dump(data_to_store, f)