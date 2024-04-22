import os
import pandas as pd

# Define the directory where the files are located
directory = r"D:\\Homework\\Endterm\\Result"

# Get a list of all files in the directory
files = os.listdir(directory)

# Initialize a list to store all data
all_data = []

# Iterate over each file
for file in files:
    # Construct the full file path
    filepath = os.path.join(directory, file)

    # Read the file into a DataFrame
    data = pd.read_csv(filepath, sep=",", engine="python")

    # Check the number of columns in the DataFrame
    if len(data.columns) == 10:
        # This is the format with two extra columns
        # We need to shift the data two columns to the right
        data[data.columns[2:]] = data[data.columns[:-2]]
        # Fill the first two columns with NaN
        data[data.columns[:2]] = None

    # Add a column to the DataFrame to store the file name
    data["file"] = file

    # Append the data to the all_data list
    all_data.append(data)

# Concatenate all the dataframes in the list
all_data = pd.concat(all_data)

# Write the all_data DataFrame to an Excel file
all_data.to_excel("all_data.xlsx", index=False)
