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

tif_dir = r"D:\\thesis\\\\00shptif"
csv_dir = r"D:\\thesis\\\\11shptif"

# Convert all tif files in the directory to csv files
tif_to_csv(tif_dir, csv_dir)
