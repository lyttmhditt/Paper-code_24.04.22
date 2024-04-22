import pandas as pd
import numpy as np

# Read the data from the Excel file
all_data = pd.read_excel("all_data.xlsx")

# It seems like there are leading and trailing whitespaces in the column names, let's strip them
all_data.columns = all_data.columns.str.strip()

# Replace empty strings with np.nan in the 'TYPE' column
all_data['TYPE'].replace('', np.nan, inplace=True)

# Split the data into two DataFrames based on whether the "TYPE" field is empty
data_with_type = all_data.dropna(subset=['TYPE'])
data_without_type = all_data[pd.isnull(all_data['TYPE'])]

# Write the data_without_type DataFrame to a new Excel file
data_without_type.to_excel("dsuma.xlsx", index=False)

# For the data with 'TYPE', calculate the sum of each column for each unique value in the "TYPE" field
sums_with_type = data_with_type.groupby('TYPE').sum(numeric_only=True)

# Write the sums_with_type DataFrame to a new Excel file named "dsumb.xlsx"
sums_with_type.to_excel("dsumb.xlsx")
