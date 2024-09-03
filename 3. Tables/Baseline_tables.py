import pandas as pd
import numpy as np
from tableone import TableOne

class DataPreprocessor:
    def __init__(self):
        pass

    def preprocess_full(self, df):
        self._convert_ASIA_to_binary(df)
        self._group_KPS(df)
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

    def _round_variables(self, df):
        df[["BMI", "B_Index", "M3_Index"]] = df[["BMI", "B_Index", "M3_Index"]].round({"BMI": 0, "B_Index": 3, "M3_Index": 3})

    def _set_data_types(self, df):
        binary_columns = ["Gender", "CCI_YN", "binary_ASIA", "Tumor C-level", "Tumor T-level", "Tumor L-level", "Tumor S-level",
                          "Organ", "Path_Fract", "Prev_Syst", "Opioid", "3_months"]
        integer_columns = ["Age", "BMI", "Katagiri_Group", "grouped_KPS", "Nr_Spine_Met", "CHI+RT", "B_Mob", "B_Sel", "B_Usu",
                           "B_Dis", "B_Anx"]
        continuous_columns = ["B_Index", "M3_Index"]

        df[binary_columns] = df[binary_columns].astype(bool)
        df[integer_columns] = df[integer_columns].astype("int8")
        df[continuous_columns] = df[continuous_columns].astype("float64")

# Baseline table
df = pd.read_excel("...")
df["MCID_Result"] = np.where(df["M3_Index"].isna() | df["B_Index"].isna(), np.nan, 
                             (df["M3_Index"] - df["B_Index"] >= 0.08).astype(int))

# Grouped variables
df = DataPreprocessor().preprocess_group(df)

features = ["Age", "BMI", "Gender", "CCI_YN", "grouped_KPS", "binary_ASIA",
            "Functional_Stat_1", "Pre_Chem",
            "Tumor C-level", "Tumor T-level", "Tumor L-level", "Tumor S-level", 
            "Katagiri_Group", "Visceral", "Brain", "Path_Fract", "Prev_Syst", "Opioid",
            "CHI+RT",
            "B_Mob", "B_Sel", "B_Usu", "B_Dis", "B_Anx", "B_Index",
            "M3_Mob", "M3_Sel", "M3_Usu", "M3_Dis", "M3_Anx", "M3_months", "M3_Index", "3_months", "12_months", "MCID_Result"]

categorical_features = ["Gender", "CCI_YN", "grouped_KPS", "binary_ASIA",
                        "Functional_Stat_1", "Pre_Chem",
                        "Tumor C-level", "Tumor T-level", "Tumor L-level", "Tumor S-level", 
                        "Katagiri_Group", "Visceral", "Brain", "Path_Fract", "Prev_Syst", "Opioid",
                        "CHI+RT","B_Mob", "B_Sel", "B_Usu", "B_Dis", "B_Anx",
                        "M3_Mob", "M3_Sel", "M3_Usu", "M3_Dis", "M3_Anx", "3_months", "12_months", "MCID_Result"]
          
nonnormal_features = ["Age", "BMI", "B_Index", "M3_months", "M3_Index"]
decimals = {"B_Index": 3, "M3_Index": 3}
baseline_table = TableOne(df, columns=features, categorical=categorical_features, nonnormal=nonnormal_features, pval=False, decimals=decimals)
print(baseline_table.tabulate(tablefmt = "fancy_grid"))

baseline_table.to_excel("...")
##################################################################################################################################################
df["B_Index_complete"] = np.where(df['B_Index'].isnull(), 0, 1)
df["M3_Index_complete"] = np.where(df['M3_Index'].isnull(), 0, 1)

groupby = ["B_Index_complete"]
EQ5D_B_comparison = TableOne(df, columns=features, categorical=categorical_features, 
                                 groupby=groupby, nonnormal=nonnormal_features, pval=True, htest_name=True, decimals=decimals)
print(EQ5D_B_comparison.tabulate(tablefmt = "fancy_grid"))
EQ5D_B_comparison.to_excel("...")

groupby = ["M3_Index_complete"]
EQ5D_M3_comparison = TableOne(df, columns=features, categorical=categorical_features, 
                                 groupby=groupby, nonnormal=nonnormal_features, pval=True, htest_name=True, decimals=decimals)
print(EQ5D_M3_comparison.tabulate(tablefmt = "fancy_grid"))
EQ5D_M3_comparison.to_excel("...")

##################################################################################################################################################
X_train = pd.read_excel(r"...")
X_test = pd.read_excel(r"...")
y_train = pd.read_excel(r"...")
y_test = pd.read_excel(r"...")

# Concatenate y_train and y_test with X_train and X_test, respectively
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Combine the train and test data into one DataFrame
combined_data = DataPreprocessor().preprocess_group(pd.concat([train_data, test_data], ignore_index=True, axis=0))

groupby = ["Test"]
train_test_comparison = TableOne(combined_data, columns=features, categorical=categorical_features, 
                                 groupby=groupby, nonnormal=nonnormal_features, pval=True, htest_name=True, decimals=decimals)
print(train_test_comparison.tabulate(tablefmt = "fancy_grid"))

train_test_comparison.to_excel("...")

groupby = ["Gender"]
train_test_comparison = TableOne(combined_data, columns=features, categorical=categorical_features, 
                                 groupby=groupby, nonnormal=nonnormal_features, pval=True, htest_name=True, decimals=decimals)
print(train_test_comparison.tabulate(tablefmt = "fancy_grid"))
