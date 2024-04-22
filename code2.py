import os
import pandas as pd
import rasterio
from glob import glob

def tif_to_csv(tif_dir, csv_dir):  # Convert tif files to csv files, tif directory, csv directory
    # Get all tif files in the directory
    tif_files = glob(os.path.join(tif_dir, "*.tif"))

    for tif_file in tif_files:
        try:
            # Read tif file
            with rasterio.open(tif_file) as src:
                meta = src.meta

            # Create csv file path
            base_name = os.path.basename(tif_file)
            csv_file = os.path.join(csv_dir, os.path.splitext(base_name)[0] + ".csv")

            # Prepare the required information
            info = [tif_file, 'x', 999, meta['height'], meta['width'], meta['count'], -9999, 'IDF_GeoTIFF']

            # Save the information to csv file
            pd.DataFrame([info]).to_csv(csv_file, index=False, header=False)

            print(f"{tif_file} has been successfully converted to {csv_file}")
        except Exception as e:
            print(f"Error occurred while processing {tif_file}: {str(e)}")

def frg_cmd(model, scripts_dir, output_dir):  # Model path, script directory, output directory
    csv_scripts = glob(os.path.join(scripts_dir, "*.csv"))  # Grab all csv files in the directory
    for csv_script in csv_scripts:
        # Create a unique output file name for each csv file
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(csv_script))[0] + ".class")
        print("=====================================================")
        print(f"model: {model}")
        print(f"script: {csv_script}")
        print(f"output: {output_file}")
        print("\\n")
        os.system(f"frg -m {model} -b {csv_script} -o {output_file}")  # Command line execution 2
        if os.path.exists(output_file):  # Check if the output file exists before reading it
            read_csv_script = pd.read_csv(csv_script)  # Read script file
            read_output_file = pd.read_csv(output_file)  # Read output file
            read_csv_script = pd.DataFrame(read_csv_script)
            read_output_file = pd.DataFrame(read_output_file)

            print("\\n")
            print(f"{output_file} is completed!")
        else:
            print(f"Error: {output_file} was not created.")
        print("=====================================================")

tif_dir = r"D:\thesis\Extract_tif13"
csv_dir = r"D:\thesis\13csv"
model = r"D:\Homework\Endterm\fragproject\projectall.fca"
scripts_dir = csv_dir
output_dir = r"D:\thesis\13resullt"

# Convert all tif files in the directory to csv files
tif_to_csv(tif_dir, csv_dir)

# Run the frg_cmd function
frg_cmd(model, scripts_dir, output_dir)
