import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        pass

    def preprocess_full(self, df):
        self._convert_ASIA_to_binary(df)
        self._group_KPS(df)
        self._group_Age(df)
        self._group_BMI(df)
        self._round_variables(df)
        self._set_data_types(df)
        return df
    
    def preprocess_group(self, df):
        self._convert_ASIA_to_binary(df)
        self._group_KPS(df)
        return df

    def _convert_ASIA_to_binary(self, df):
        df['binary_ASIA'] = df['ASIA'].apply(lambda x: 0 if x <= 3 else (1 if pd.notnull(x) else np.nan))

    def _group_KPS(self, df):
        df['grouped_KPS'] = pd.cut(df['KPS'], bins=[0, 40, 70, 100], labels=[0, 1, 2], right=True)
        
    def _group_Age(self, df):
        df['grouped_Age'] = pd.cut(df['Age'], bins=[0, 60, 80, 100], labels=[0, 1, 2], right=True)
        
    def _group_BMI(self, df):
        df['grouped_BMI'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 80], labels=[0, 1, 2, 3], right=False)

    def _round_variables(self, df):
        df.round({"BMI": 0, "B_Index": 3, "M3_Index": 3})

    def _set_data_types(self, df):
        binary_columns = ["Gender", "CCI_YN", "binary_ASIA", "Tumor C-level", "Tumor T-level", "Tumor L-level", "Tumor S-level",
                          "Functional_Stat_1", "Visceral", "Brain", "Path_Fract", "Prev_Syst", "Pre_Chem", "Opioid", "3_months", "12_months"]
        integer_columns = ["Age", "BMI", 'grouped_Age', 'grouped_BMI', "Katagiri_Group", "grouped_KPS", "CHI+RT", "B_Mob", "B_Sel", "B_Usu",
                           "B_Dis", "B_Anx"]
        continuous_columns = ["KPS", "B_Index", "M3_Index"]

        df[binary_columns] = df[binary_columns].astype(bool)
        df[integer_columns] = df[integer_columns].astype("int8")
        df[continuous_columns] = df[continuous_columns].astype("float64")

# Load the existing variables
with open(r'...', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

# Load data
df = pd.read_excel(r"...", usecols = data["all_columns"])
df["MCID_Result"] = (df["M3_Index"] - df["B_Index"] >= 0.08).astype(int)

# Separate the features (X) and the target variable (y)
X = df[data["features"]]
y = df["MCID_Result"]

# Perform the stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, stratify = y, random_state = 42)

# Save groups for comparison table
X_train["Test"] = 0
X_test["Test"] = 1

preprocessor = DataPreprocessor()
X_train = preprocessor.preprocess_full(X_train)
X_test = preprocessor.preprocess_full(X_test)
y_train.astype(bool)
y_test.astype(bool)

X_train.to_excel(r"...", index=False)
y_train.to_excel(r"...", index=False)
X_test.to_excel(r"...", index=False)
y_test.to_excel(r"...", index=False)

# Put dataframes into a dictionary
data_dict = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test
}

# Save the dictionary to a pickle file
with open(r'...', 'wb') as file:
    pickle.dump(data_dict, file)

pred_features = ["Age", "BMI", 'grouped_Age', 'grouped_BMI', "Gender", "CCI_YN", "KPS", "grouped_KPS", "binary_ASIA", "Functional_Stat_1",
                 "Tumor C-level", "Tumor T-level", "Tumor L-level", "Tumor S-level",
                 "Katagiri_Group", "Visceral", "Brain", "Path_Fract", "Prev_Syst", "Pre_Chem", "Opioid",
                 "B_Mob", "B_Sel", "B_Usu", "B_Dis", "B_Anx", "B_Index"]

pred_cat = ['grouped_Age', 'grouped_BMI', "Gender", "CCI_YN", "grouped_KPS", "binary_ASIA", "Functional_Stat_1",
                                "Tumor C-level", "Tumor T-level", "Tumor L-level", "Tumor S-level",
                                "Katagiri_Group", "Path_Fract", "Prev_Syst", "Pre_Chem", "Opioid",
                                "B_Mob", "B_Sel", "B_Usu", "B_Dis", "B_Anx"]
          
pred_num = ["Age", "BMI", "KPS", "B_Index"]

data["pred_features"] = pred_features
data["pred_cat"] = pred_cat
data["pred_num"] = pred_num

with open(r'...', 'wb') as pickle_file:
    pickle.dump(data, pickle_file)