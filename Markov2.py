import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from PIL import Image

def transition_matrix(Start_LUCC_image, Mid_LUCC_image, End_LUCC_image):
    Start_LUCC_image = Image.open(Start_LUCC_image)
    Mid_LUCC_image = Image.open(Mid_LUCC_image)
    End_LUCC_image = Image.open(End_LUCC_image)
    lucc_start_array = np.asarray(Start_LUCC_image).flatten()
    lucc_mid_array = np.asarray(Mid_LUCC_image).flatten()
    lucc_end_array = np.asarray(End_LUCC_image).flatten()
    transition_area_1 = confusion_matrix(lucc_start_array, lucc_mid_array)
    transition_area_2 = confusion_matrix(lucc_mid_array, lucc_end_array)
    trans_matr_1 = transition_area_1[:len(transition_area_1), :len(transition_area_1)]
    trans_matr_2 = transition_area_2[:len(transition_area_2), :len(transition_area_2)]
    print("transition_matrix 1:", "\n", trans_matr_1)
    print("transition_matrix 2:", "\n", trans_matr_2)
    return trans_matr_1, trans_matr_2

def markov(trans_matr_1, trans_matr_2, Start_year, Mid_year, End_year, Pred_year):
    n = int((Pred_year - End_year) / (Mid_year - Start_year))
    sum_Start_year = trans_matr_1.sum(axis=1)
    sum_Mid_year = trans_matr_2.sum(axis=1)
    whole_area = trans_matr_1.sum()
    P_Mid_year = sum_Mid_year / trans_matr_1.sum()
    # Calculate the transition probabilities for the second order Markov chain
    Ptrans0_1_2 = np.empty((len(sum_Start_year), len(sum_Start_year), len(sum_Start_year)))
    for i in range(len(sum_Start_year)):
        for j in range(len(sum_Start_year)):
            Ptrans0_1_2[i, j] = trans_matr_1[i, j] / sum_Start_year[i]

    # Use the second order transition probabilities to predict the land use
    E = np.identity(len(sum_Start_year))
    for i in range(n):
        E = np.dot(E, np.dot(Ptrans0_1, Ptrans0_1_2))
    Ptrans = E
    P_Pred_year = np.dot(P_Mid_year, Ptrans)
    # Convert pixel count to area in square kilometers
    Pred_year_area = np.array(np.around((P_Pred_year * whole_area / 1000000) * 900, decimals=7))
    np.set_printoptions(formatter={'float': '{: 0.7f}'.format})
    print("转移概率矩阵:", "\n", np.around(Ptrans, decimals=7))
    print("预测年份状态矩阵:", "\n", np.around(P_Pred_year, decimals=7))
    print("预测年份各地类面积 (in square kilometers):", Pred_year, "年\n", Pred_year_area)
    return Ptrans, P_Pred_year, Pred_year_area

def Save(Pred_year, Ptrans, P_Pred_year, Pred_year_area):
    outpath = r"D:\thesis\Markov\\" + str(Pred_year) + ".txt"
    with open(outpath, 'w') as f:
        f.write('%d' % Pred_year)
        f.write(":\n")
        f.write("转移概率矩阵:\n")
        np.savetxt(f, np.round(Ptrans, 7), delimiter="\t", fmt="%.7f")
        f.write("预测年份状态矩阵:\n")
        np.savetxt(f, np.round(P_Pred_year, 7), delimiter="\t", fmt="%.7f")
        f.write("预测年份各地类面积 (in pixel count):\n")
        np.savetxt(f, Pred_year_area, delimiter="\t", fmt="%s")
    f.close()

def backtest():
    # Read the images for 2013, 2017 and 2021
    lucc_2013_img = r"D:\thesis\tif2013.tif"
    lucc_2017_img = r"D:\thesis\tif2017.tif"
    lucc_2021_img = r"D:\thesis\tif2021.tif"

    # Calculate the transition matrix for 2013 to 2017 and 2017 to 2021
    trans_matr_1, trans_matr_2 = transition_matrix(lucc_2013_img, lucc_2017_img, lucc_2021_img)

    # Use the Markov model to predict the land use for 2021
    Ptrans, P_Pred_year, Pred_year_area = markov(trans_matr_1, trans_matr_2, Start_year=2013, Mid_year=2017, End_year=2021, Pred_year=2021)

    # Calculate the actual land use distribution for 2021
    lucc_2021_image = Image.open(lucc_2021_img)
    lucc_2021_array = np.asarray(lucc_2021_image).flatten()
    # Ensure all pixel values are non-negative
    lucc_2021_array[lucc_2021_array < 0] = 0
    actual_2021_area = np.bincount(lucc_2021_array, minlength=6)
    # Convert pixel count to area in square kilometers
    actual_2021_area = np.around((actual_2021_area / 1000000) * 900, decimals=7)

    # Calculate the Root Mean Squared Relative Error
    rmsre = np.sqrt(np.mean(((Pred_year_area - actual_2021_area) / (actual_2021_area + 1e-10)) ** 2))
    print("Root Mean Squared Relative Error:", rmsre)

    # Save the results
    Save(2021, Ptrans, P_Pred_year, Pred_year_area)
if __name__ == '__main__':
    backtest()
