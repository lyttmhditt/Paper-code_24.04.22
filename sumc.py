import pandas as pd

# Read the data from the Excel file
all_data = pd.read_excel("o.xlsx")

# Calculate the sum of each column
column_sums = all_data.sum()

# Create a new DataFrame to store the sums
sum_df = pd.DataFrame(column_sums).transpose()

# Write the sum_df DataFrame to a new Excel file
sum_df.to_excel("sum.xlsx", index=False)
