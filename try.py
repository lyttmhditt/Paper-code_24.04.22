import numpy as np
from sklearn.preprocessing import normalize
import rasterio

# 读取你的初始土地利用图像
with rasterio.open('D:\\thesis\\Endterm\\Markov\\hebishi2017.tif') as src:
    img1 = src.read(1).astype(float)  # 读取第一层数据并转换为浮点数
    transform = src.transform  # 获取地理变换信息
    crs = src.crs  # 获取坐标参考系统（CRS）信息

with rasterio.open('D:\\thesis\\Endterm\\Markov\\hebishi2020.tif') as src:
    img2 = src.read(1).astype(float)  # 读取第一层数据并转换为浮点数

# 确保两幅图像具有相同的尺寸
assert img1.shape == img2.shape, "The two images must have the same size."

# 将无数据的像素值替换为np.nan
img1[img1 == -9999] = np.nan
img2[img2 == -9999] = np.nan

# 统计每种像素类型的数量
unique, counts = np.unique(img1[~np.isnan(img1)], return_counts=True)
pixel_counts = dict(zip(unique, counts))

# 根据像素类型的数量，从大到小进行排序
sorted_pixel_types = sorted(pixel_counts, key=pixel_counts.get, reverse=True)

# 限制唯一值的数量为5
sorted_pixel_types = sorted_pixel_types[:5]

# 计算转移矩阵
trans_mat = np.zeros((len(sorted_pixel_types), len(sorted_pixel_types)))
for i in range(1, img1.shape[0]-1):
    for j in range(1, img1.shape[1]-1):
        if np.isnan(img1[i, j]) or np.isnan(img2[i, j]):
            continue
        neighbors = [img1[x, y] for x in range(i-1, i+2) for y in range(j-1, j+2) if (x, y) != (i, j)]
        for neighbor in neighbors:
            if np.isnan(neighbor):
                continue
            trans_mat[sorted_pixel_types.index(img1[i, j]), sorted_pixel_types.index(neighbor)] += 1
trans_mat = normalize(trans_mat, axis=1, norm='l1')

# 使用马尔科夫链预测未来的土地利用图像
future_img = img2.copy()
for _ in range(10):  # 预测10个时间步
    next_img = future_img.copy()
    for i in range(1, future_img.shape[0]-1):
        for j in range(1, future_img.shape[1]-1):
            if np.isnan(future_img[i, j]):
                continue
            neighbors = [future_img[x, y] for x in range(i-1, i+2) for y in range(j-1, j+2) if (x, y) != (i, j)]
            neighbor_counts = np.bincount([sorted_pixel_types.index(neighbor) for neighbor in neighbors if not np.isnan(neighbor)], minlength=len(sorted_pixel_types))
            transition_probs = np.sum(trans_mat[sorted_pixel_types.index(future_img[i, j])] * neighbor_counts)
            next_img[i, j] = sorted_pixel_types[np.argmax(transition_probs)]
    future_img = next_img

    # 将预测的土地利用图像保存到指定的文件夹
    output_path = f'D:\\thesis\\Endterm\\Markov\\future_img_{_+1}.tif'
    with rasterio.open(output_path, 'w', driver='GTiff', height=future_img.shape[0], width=future_img.shape[1], count=1, dtype=str(future_img.dtype), crs=crs, transform=transform) as dst:
        dst.write(future_img, 1)
