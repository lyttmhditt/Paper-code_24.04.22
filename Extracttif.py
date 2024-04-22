from osgeo import gdal
import glob
import numpy as np
import csv

# 计算栅格图像均值
def calc_grid_mean(tiffile, csvpath):
    open_tif = gdal.Open(tiffile) # 打开tif文件
    # 打开栅格图像的波段
    band = open_tif.GetRasterBand(1).ReadAsArray()
    count_value = [] # 存放像元值的数组
    # 遍历图像矩阵，剔除异常值，读取有效值
    for i in range(len(band)):
        for j in range(len(band[i])):
            if band[i][j] >= -1000:
                count_value.append(band[i][j]) # 将有效值放入新数组
    # 把文件名和参数写入csv
    mycsv = open(str(csvpath), 'a', newline='') # 新建并打开csv
    csv_write = csv.writer(mycsv, dialect='excel')
    tif_info = (str(tif[-18:]), float(np.mean(count_value))) # 把文件名称+均值整合为一行
    csv_write.writerow(tif_info) # 把文件名称+均值写入csv
    print(f'{str(tif[-18:])}, {np.mean(count_value)}') # 在终端中打印出文件名称+均值


csv_path = r'csv.csv' # csv输出路径
tif_dir = glob.glob(r'文件夹\*.tif') # tif目录

for tif in tif_dir: # 遍历文件夹
    calc_grid_mean(tif, csv_path) # 计算均值
