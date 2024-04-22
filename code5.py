import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import fractional_matrix_power
import rasterio
from PIL import Image

# 读取你的初始土地利用图像
with rasterio.open('D:\\thesis\\Endterm\\Markov\\hebishi2020.tif') as src:
    img = src.read(1)  # 读取第一层数据
    transform = src.transform  # 获取地理变换信息
    crs = src.crs  # 获取坐标参考系统（CRS）信息

# 将无数据的像素值替换为0
img[img == -9999] = 0

# 计算转移矩阵
trans_mat = np.zeros((6, 6))
for i in range(img.shape[0]-1):
    for j in range(img.shape[1]-1):
        trans_mat[img[i, j], img[i+1, j+1]] += 1
trans_mat = normalize(trans_mat, axis=1, norm='l1')

# 使用马尔科夫链预测未来的土地利用图像
future_img = img.copy()
for _ in range(2):  # 预测2个时间步
    for i in range(future_img.shape[0]):
        for j in range(future_img.shape[1]):
            future_img[i, j] = np.random.choice(6, p=trans_mat[future_img[i, j]])
    # 将预测的土地利用图像保存到指定的文件夹
    output_path = f'D:\\thesis\\Endterm\\Markov\\future_img_{_+1}.tif'
    with rasterio.open(output_path, 'w', driver='GTiff', height=future_img.shape[0], width=future_img.shape[1], count=1, dtype=str(future_img.dtype), crs=crs, transform=transform) as dst:
        dst.write(future_img, 1)
