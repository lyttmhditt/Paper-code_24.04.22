import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from PIL import Image
from osgeo import gdal, gdal_array

def array_to_raster(array, reference_tif, output_tif):
    dataset = gdal.Open(reference_tif)
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    data_type = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_tif, array.shape[1], array.shape[0], 1, data_type)
    out_dataset.SetGeoTransform(geotransform)
    out_dataset.SetProjection(projection)
    out_dataset.GetRasterBand(1).WriteArray(array)
    out_dataset.FlushCache()

def transition_matrix(Start_LUCC_image, End_LUCC_image):
    Start_LUCC_image = Image.open(Start_LUCC_image)
    End_LUCC_image = Image.open(End_LUCC_image)
    lucc_start_array = np.asarray(Start_LUCC_image).flatten()
    lucc_end_array = np.asarray(End_LUCC_image).flatten()
    transition_area = confusion_matrix(lucc_start_array, lucc_end_array)
    trans_matr = transition_area[:len(transition_area), :len(transition_area)]
    print("transition_matrix:", "\n", trans_matr)
    return trans_matr

def markov(trans_matr, Start_year, End_year, Pred_year):
    sum_Start_year = trans_matr.sum(axis=1)
    sum_End_year = trans_matr.sum(axis=0)
    whole_area = trans_matr.sum()
    P_End_year = sum_End_year / trans_matr.sum()
    Ptrans0 = np.empty(trans_matr.shape)
    for i in range(len(sum_Start_year)):
        Ptrans0[i] = trans_matr[i] / sum_Start_year[i]
    n = int((Pred_year - End_year) / (End_year - Start_year))
    E = np.identity(len(sum_Start_year))
    for i in range(n):
        E = np.dot(E, Ptrans0)
    Ptrans = E
    P_Pred_year = np.dot(P_End_year, Ptrans)
    Pred_year_area = np.array(np.around(P_Pred_year * whole_area * 30 * 30 / 1000000, decimals=7), dtype=float)
    np.set_printoptions(formatter={'float': '{: 0.7f}'.format})
    print("转移概率矩阵:", "\n", np.around(Ptrans, decimals=7))
    print("预测年份状态矩阵:", "\n", np.around(P_Pred_year, decimals=7))
    print("预测年份各地类面积 (in square kilometers):", Pred_year, "年\n", Pred_year_area)
    return Ptrans, P_Pred_year, Pred_year_area[1:]

def Save(Pred_year, Pred_year_area, df):
    df.loc[Pred_year] = Pred_year_area
    return df

if __name__ == '__main__':
    lucc_2017_img = r"D:\thesis\tif2017.tif"
    lucc_2021_img = r"D:\thesis\tif2021.tif"
    Start_year = 2017
    End_year = 2021
    Pred_year = 2025
    last_Pred_year_area = None
    df = pd.DataFrame(columns=['water', 'settlement', 'land', 'grassland', 'bareland'])
    df.loc['2021'] = [27.5841, 460.8189, 1334.5263, 47.9007, 271.5336]
    while Pred_year <= 2045:
        trans_matr = transition_matrix(lucc_2017_img, lucc_2021_img)
        Ptrans, P_Pred_year, Pred_year_area = markov(trans_matr, Start_year, End_year, Pred_year)
        df = Save(str(Pred_year), Pred_year_area, df)
        if last_Pred_year_area is not None and np.allclose(last_Pred_year_area, Pred_year_area):
            break
        last_Pred_year_area = Pred_year_area
        Pred_year += 4
        lucc_2017_img = lucc_2021_img
        lucc_2021_img = r"D:\thesis\Markov\tif" + str(Pred_year) + ".tif"
        # Convert the predicted area to a 2D numpy array
        pred_array = np.array([Pred_year_area])
        # Save the numpy array as a tif file
        array_to_raster(pred_array, lucc_2017_img, lucc_2021_img)
    df = df.transpose()
    df.to_excel(r"D:\thesis\Markov\prediction_areas.xlsx", index=True)
