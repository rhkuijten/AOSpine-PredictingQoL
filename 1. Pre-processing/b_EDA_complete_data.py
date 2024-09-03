import pandas as pd
import pickle
from ydata_profiling import ProfileReport

# All columnns used from the database
all_columns = ['NEW_ID','Age','BMI','Gender',
           'CCI_YN','KPS','Functional_Stat_1','ASIA',
           'Katagiri_Primary', 'Katagiri_Group','Tumor C-level','Tumor T-level','Tumor L-level','Tumor S-level',
           'Visceral','Brain','Path_Fract',
           'CHI+RT', 'Prev_Syst', 'Pre_Chem','Opioid',
           'B_Mob', 'B_Sel','B_Usu','B_Dis','B_Anx','B_Index',
           'M3_Mob','M3_Sel','M3_Usu','M3_Dis','M3_Anx','M3_months', "M3_Index", "3_months", "12_months"]

features = ['Age','BMI','Gender',
           'CCI_YN','KPS','Functional_Stat_1','ASIA',
           'Katagiri_Primary', 'Katagiri_Group','Tumor C-level','Tumor T-level','Tumor L-level','Tumor S-level',
           'Visceral','Brain','Path_Fract',
           'CHI+RT', 'Prev_Syst', 'Pre_Chem','Opioid',
           'B_Mob', 'B_Sel','B_Usu','B_Dis','B_Anx','B_Index',
           'M3_Mob','M3_Sel','M3_Usu','M3_Dis','M3_Anx','M3_months', "M3_Index", "3_months", "12_months"]

cat_features = ['Gender','CCI_YN','KPS','Functional_Stat_1','ASIA',
               'Katagiri_Primary', 'Katagiri_Group','Tumor C-level','Tumor T-level','Tumor L-level','Tumor S-level',
               'Visceral','Brain','Path_Fract',
               'CHI+RT', 'Prev_Syst', 'Pre_Chem','Opioid',
               'B_Mob', 'B_Sel','B_Usu','B_Dis','B_Anx',
               'M3_Mob','M3_Sel','M3_Usu','M3_Dis','M3_Anx', "3_months", "12_months"]

num_features = ['Age','BMI','B_Index','M3_months', "M3_Index"]

# Importing data
df = pd.read_excel("...", usecols = features)

# Building complete data EDA
ProfileReport(df).to_file("...")

# Initializing and saving the all_columns variable
with open('my_variables.pkl', 'wb') as pickle_file:
    pickle.dump({'all_columns': all_columns, "features": features, "cat_columns": cat_features, "num_columns" : num_features}, pickle_file)
