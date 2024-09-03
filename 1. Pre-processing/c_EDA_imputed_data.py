import pandas as pd
import pickle
from ydata_profiling import ProfileReport

# Load the existing variables
with open('my_variables.pkl', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

df = pd.read_excel("...", usecols = data["all_columns"])
df["MCID_Result"] = (df["M3_Index"] - df["B_Index"] >= 0.08).astype(int)

# Save the existing variables
with open('my_variables.pkl', 'wb') as pickle_file:
    pickle.dump(data, pickle_file)

ProfileReport(df).to_file("...")

df.to_excel("...")