import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from PIL import Image

# 定义函数计算转移矩阵
def transition_matrix(Start_LUCC_image, End_LUCC_image):
    # 打开开始和结束年份的土地利用图像
    Start_LUCC_image = Image.open(Start_LUCC_image)
    End_LUCC_image = Image.open(End_LUCC_image)
    # 将图像转换为一维数组
    lucc_start_array = np.asarray(Start_LUCC_image).flatten()
    lucc_end_array = np.asarray(End_LUCC_image).flatten()
    # 计算混淆矩阵，即转移矩阵
    transition_area = confusion_matrix(lucc_start_array, lucc_end_array)
    # 保留转移矩阵的前n行和前n列
    trans_matr = transition_area[:len(transition_area), :len(transition_area)]
    print("transition_matrix:", "\n", trans_matr)
    return trans_matr

# 定义函数进行马尔可夫链预测
def markov(trans_matr, Start_year, End_year, Pred_year):
    # 计算转移矩阵的行和列之和
    sum_Start_year = trans_matr.sum(axis=1)
    sum_End_year = trans_matr.sum(axis=0)
    # 计算转移矩阵的总和
    whole_area = trans_matr.sum()
    # 计算结束年份的概率分布
    P_End_year = sum_End_year / trans_matr.sum()
    # 初始化转移概率矩阵
    Ptrans0 = np.empty(trans_matr.shape)
    # 计算转移概率矩阵
    for i in range(len(sum_Start_year)):
        Ptrans0[i] = trans_matr[i] / sum_Start_year[i]
    # 计算预测年份与结束年份之间的时间间隔
    n = int((Pred_year - End_year) / (End_year - Start_year))
    # 初始化单位矩阵
    E = np.identity(len(sum_Start_year))
    # 计算n次转移后的转移概率矩阵
    for i in range(n):
        E = np.dot(E, Ptrans0)
    Ptrans = E
    # 计算预测年份的概率分布
    P_Pred_year = np.dot(P_End_year, Ptrans)
    # 将像素计数转换为平方公里面积
    Pred_year_area = np.array(np.around(P_Pred_year * whole_area * 30 * 30 / 1000000, decimals=7), dtype=float)
    np.set_printoptions(formatter={'float': '{: 0.7f}'.format})
    print("转移概率矩阵:", "\n", np.around(Ptrans, decimals=7))
    print("预测年份状态矩阵:", "\n", np.around(P_Pred_year, decimals=7))
    print("预测年份各地类面积 (in square kilometers):", Pred_year, "年\n", Pred_year_area)
    return Ptrans, P_Pred_year, Pred_year_area[1:]  # 忽略第一个面积值

# 定义函数保存预测结果
def Save(Pred_year, Pred_year_area, df):
    df.loc[Pred_year] = Pred_year_area
    return df

if __name__ == '__main__':
    # 定义开始和结束年份的土地利用图像路径
    lucc_2017_img = r"D:\thesis\tif2017.tif"
    lucc_2021_img = r"D:\thesis\tif2021.tif"
    # 计算转移矩阵
    trans_matr = transition_matrix(lucc_2017_img, lucc_2021_img)
    Start_year = 2017
    End_year = 2021
    Pred_year = 2025
    last_Pred_year_area = None
    # 定义数据框的列名
    df = pd.DataFrame(columns=['water', 'settlement', 'land', 'grassland', 'bareland'])
    # 添加2021年的数据到数据框
    df.loc['2021'] = [27.5841, 460.8189, 1334.5263, 47.9007, 271.5336]
    while True:
        # 进行马尔可夫链预测
        Ptrans, P_Pred_year, Pred_year_area = markov(trans_matr, Start_year, End_year, Pred_year)
        # 保存预测结果
        df = Save(str(Pred_year), Pred_year_area, df)
        # 如果预测结果收敛，则停止预测
        if last_Pred_year_area is not None and np.allclose(last_Pred_year_area, Pred_year_area):
            break
        last_Pred_year_area = Pred_year_area
        Pred_year += 4
    # 转置数据框
    df = df.transpose()
    # 保存数据框到Excel文件
    df.to_excel(r"D:\thesis\Markov\prediction_areas.xlsx", index=True)
