import shap
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from utils_evaluation import bootstrap_model_performance
import pickle

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
    
features = ['B_Index', 'Katagiri_Group', 'Brain', 'Opioid', 'grouped_KPS']
features_num = ['B_Index']

power_and_scale_transformer = CustomPowerTransformer(columns=features_num, all_features=features)

# Load the existing variables
with open("my_variables.pkl", "rb") as pickle_file:
    data = pickle.load(pickle_file)

# Load the processed datasets
with open("processed_data.pickle", "rb") as pickle_file:
    processed_data = pickle.load(pickle_file)
  
with open(r"...", 'rb') as f:
    model = pickle.load(f)

features = ['B_Index', 'Katagiri_Group', 'Brain', 'Opioid', 'grouped_KPS']
features_num = ['B_Index']

X_test = processed_data["X_test"]
y_test = processed_data["y_test"].copy()

# Get indices for Female 0 and Male 1
indices_gender_0 = X_test[X_test['Gender'] == 0].index
indices_gender_1 = X_test[X_test['Gender'] == 1].index

# Filter X_test and y_test for Gender 0
X_test_gender_0 = X_test.loc[indices_gender_0].filter(items=features)
y_test_gender_0 = y_test.loc[indices_gender_0]

# Filter X_test and y_test for Gender 1
X_test_gender_1 = X_test.loc[indices_gender_1].filter(items=features)
y_test_gender_1 = y_test.loc[indices_gender_1]

auc, _, _, brier, _ = bootstrap_model_performance(model, X_test[features], y_test)
auc_female, _, _, brier_female, _ = bootstrap_model_performance(model, X_test_gender_0, y_test_gender_0)
auc_male, _, _, brier_male, _ = bootstrap_model_performance(model, X_test_gender_1, y_test_gender_1)

gender_df = pd.DataFrame({
    'Gender': ['Female', 'Male'],
    'N': [len(X_test_gender_0), 
          len(X_test_gender_1)],
    'AUC Score': [f"{auc_female[0]} ({auc_female[1]} - {auc_female[2]})", 
                  f"{auc_male[0]} ({auc_male[1]} - {auc_male[2]})"],
    'Brier Score': [f"{brier_female[0]} ({brier_female[1]} - {brier_female[2]})",
                    f"{brier_male[0]} ({brier_male[1]} - {brier_male[2]})"],
    'MCID_count' : [y_test_gender_0["MCID_Result"].sum(),
                    y_test_gender_1["MCID_Result"].sum()]
})

gender_df.to_excel(r"...")

# Define bins for age groups
bins = [40, 60, 80, 100]
labels = ['40-60', '60-80', '80-100']
X_test['grouped_Age'] = pd.cut(X_test['Age'], bins=bins, labels=labels, right=False)

# Filter for each age group and drop the Age and Age Group columns
age_groups = {}
for label in labels:
    indices = X_test[X_test['grouped_Age'] == label].index
    X_group = X_test.loc[indices].filter(items=features)
    y_group = y_test.loc[indices]
    age_groups[label] = (X_group, y_group)

auc_40_60, _, _, brier_40_60, _ = bootstrap_model_performance(model, age_groups['40-60'][0], age_groups['40-60'][1])
auc_60_80, _, _, brier_60_80, _ = bootstrap_model_performance(model, age_groups['60-80'][0], age_groups['60-80'][1])
auc_80_100, _, _, brier_80_100, _ = bootstrap_model_performance(model, age_groups['80-100'][0], age_groups['80-100'][1])

age_df = pd.DataFrame({
    'Age Group': ['40-60', '60-80', '80-100'],
    'N' : [len(age_groups['40-60'][0]), 
           len(age_groups['60-80'][0]), 
           len(age_groups['80-100'][0])],
    'AUC Score': [f"{auc_40_60[0]} ({auc_40_60[1]} - {auc_40_60[2]})", 
                  f"{auc_60_80[0]} ({auc_60_80[1]} - {auc_60_80[2]})", 
                  f"{auc_80_100[0]} ({auc_80_100[1]} - {auc_80_100[2]})"],
    'Brier Score': [f"{brier_40_60[0]} ({brier_40_60[1]} - {brier_40_60[2]})",
                    f"{brier_60_80[0]} ({brier_60_80[1]} - {brier_60_80[2]})",
                    f"{brier_80_100[0]} ({brier_80_100[1]} - {brier_80_100[2]})"],
    'MCID_count' : [age_groups['40-60'][1]["MCID_Result"].sum(),
                    age_groups['60-80'][1]["MCID_Result"].sum(),
                    age_groups['80-100'][1]["MCID_Result"].sum()]
    
})

age_df.to_excel(r"...")

group_data = {}
for group in [1, 2, 3]:
    indices = X_test[X_test['Katagiri_Group'] == group].index
    group_data[f"{group}"] = (X_test.loc[indices].filter(items=features), y_test.loc[indices])

primary_tumors = ["Hormone dependent breast cancer", "Hormone independent breast cancer", 
                 "Hormone dependent prostate cancer", "Hormone independent prostate cancer",
                 "Non-small cell lung cancer with molecularly targeted therapy", "Other lung cancer"]

indices_breast = X_test[(X_test['Katagiri_Primary'] == "Hormone dependent breast cancer") |
                        (X_test['Katagiri_Primary'] == "Hormone independent breast cancer")].index
indices_prostate = X_test[(X_test['Katagiri_Primary'] == "Hormone dependent prostate cancer") |
                        (X_test['Katagiri_Primary'] == "Hormone independent prostate cancer")].index
indices_lung= X_test[(X_test['Katagiri_Primary'] == "Non-small cell lung cancer with molecularly targeted therapy") |
                     (X_test['Katagiri_Primary'] == "Other lung cancer")].index
indices_others = X_test[(~X_test['Katagiri_Primary'].isin(primary_tumors))].index

auc_slow, _, _, brier_slow, _ = bootstrap_model_performance(model, group_data['1'][0], group_data['1'][1])
auc_moderate, _, _, brier_moderate, _ = bootstrap_model_performance(model, group_data['2'][0], group_data['2'][1])
auc_rapid, _, _, brier_rapid, _ = bootstrap_model_performance(model, group_data['3'][0], group_data['3'][1])

auc_breast, _, _, brier_breast, _ = bootstrap_model_performance(model, X_test.loc[indices_breast].filter(items=features), y_test.loc[indices_breast])
auc_prostate, _, _, brier_prostate, _ = bootstrap_model_performance(model, X_test.loc[indices_prostate].filter(items=features), y_test.loc[indices_prostate])
auc_lung, _, _, brier_lung, _ = bootstrap_model_performance(model, X_test.loc[indices_lung].filter(items=features), y_test.loc[indices_lung])
auc_others, _, _, brier_others, _ = bootstrap_model_performance(model, X_test.loc[indices_others].filter(items=features), y_test.loc[indices_others])
                                                
tumor_df = pd.DataFrame({
    'Primary tumor': ['Slow growth', 'Moderate growth', 'Rapid growth', "Breast", "Prostate", "Lung", "Others"],
    'N': [len(group_data['1'][0]), 
          len(group_data['2'][0]), 
          len(group_data['3'][0]),
          len(X_test.loc[indices_breast]),
          len(X_test.loc[indices_prostate]),
          len(X_test.loc[indices_lung]),
          len(X_test.loc[indices_others])],
    'AUC Score': [f"{auc_slow[0]} ({auc_slow[1]} - {auc_slow[2]})",
                  f"{auc_moderate[0]} ({auc_moderate[1]} - {auc_moderate[2]})",
                  f"{auc_rapid[0]} ({auc_rapid[1]} - {auc_rapid[2]})",
                  f"{auc_breast[0]} ({auc_breast[1]} - {auc_breast[2]})",
                  f"{auc_prostate[0]} ({auc_prostate[1]} - {auc_prostate[2]})",
                  f"{auc_lung[0]} ({auc_lung[1]} - {auc_lung[2]})",
                  f"{auc_others[0]} ({auc_others[1]} - {auc_others[2]})"],
    'Brier Score': [f"{brier_slow[0]} ({brier_slow[1]} - {brier_slow[2]})",
                    f"{brier_moderate[0]} ({brier_moderate[1]} - {brier_moderate[2]})",
                    f"{brier_rapid[0]} ({brier_rapid[1]} - {brier_rapid[2]})",
                    f"{brier_breast[0]} ({brier_breast[1]} - {brier_breast[2]})",
                    f"{brier_prostate[0]} ({brier_prostate[1]} - {brier_prostate[2]})",
                    f"{brier_lung[0]} ({brier_lung[1]} - {brier_lung[2]})",
                    f"{brier_others[0]} ({brier_others[1]} - {brier_others[2]})"],
    'MCID_Result' : [group_data['1'][1]["MCID_Result"].sum(),
                     group_data['2'][1]["MCID_Result"].sum(),
                     group_data['3'][1]["MCID_Result"].sum()]
})

tumor_df.to_excel(r"...")
print(tumor_df)


# Get indices for Female 0 and Male 1
indices_chi = X_test[X_test['CHI+RT'] == 0].index
indices_chi_rt = X_test[X_test['CHI+RT'] == 1].index
indices_rt = X_test[X_test['CHI+RT'] == 2].index
indices_none = X_test[X_test['CHI+RT'] == 3].index

auc_chi, _, _, brier_chi, _ = bootstrap_model_performance(model, X_test.loc[indices_chi].filter(items=features), y_test.loc[indices_chi])
auc_chi_rt, _, _, brier_chi_rt, _ = bootstrap_model_performance(model, X_test.loc[indices_chi_rt].filter(items=features), y_test.loc[indices_chi_rt])
auc_rt, _, _, brier_rt, _ = bootstrap_model_performance(model, X_test.loc[indices_rt].filter(items=features), y_test.loc[indices_rt])
auc_none, _, _, brier_none, _ = bootstrap_model_performance(model, X_test.loc[indices_none].filter(items=features), y_test.loc[indices_none])

treatment_df = pd.DataFrame({
    'Treatment': ['Surgery', 'Surgery & Radiotherapy', 'Radiotherapy', "None"],
    'N': [len(X_test.loc[indices_chi]), 
          len(X_test.loc[indices_chi_rt]), 
          len(X_test.loc[indices_rt]),
          len(X_test.loc[indices_none])],
    'AUC Score': [f"{auc_chi[0]} ({auc_chi[1]} - {auc_chi[2]})", 
                  f"{auc_chi_rt[0]} ({auc_chi_rt[1]} - {auc_chi_rt[2]})", 
                  f"{auc_rt[0]} ({auc_rt[1]} - {auc_rt[2]})", 
                  f"{auc_none[0]} ({auc_none[1]} - {auc_none[2]})"],
    'Brier Score': [f"{brier_chi[0]} ({brier_chi[1]} - {brier_chi[2]})",
                    f"{brier_chi_rt[0]} ({brier_chi_rt[1]} - {brier_chi_rt[2]})",
                    f"{brier_rt[0]} ({brier_rt[1]} - {brier_rt[2]})",
                    f"{brier_none[0]} ({brier_none[1]} - {brier_none[2]})"],
    'MCID_Result' : [y_test.loc[indices_chi]["MCID_Result"].sum(),
                     y_test.loc[indices_chi_rt]["MCID_Result"].sum(),
                     y_test.loc[indices_rt]["MCID_Result"].sum(),
                     y_test.loc[indices_none]["MCID_Result"].sum()]
})

treatment_df.to_excel(r"...")