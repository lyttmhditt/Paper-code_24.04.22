﻿import rasterio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 读取插值图像
filepaths = [
    "D:\\thesis\\新建文件夹\\Extract_Krig1shdi22211.tif",
    "D:\\thesis\\新建文件夹\\Extract_Krig1shei22211.tif"
]  # 你的文件路径列表

# 读取图像并获取它们的形状
shapes = [rasterio.open(fp).read(1).shape for fp in filepaths]

# 检查图像是否具有相同的形状
if shapes[0] != shapes[1]:
    print("图像形状不一致，需要调整。")

    # 获取最小的形状
    min_shape = min(shapes, key=lambda x: x[0] * x[1])

    # 调整图像的大小以匹配最小的形状
    arrays = [rasterio.open(fp).read(1)[:min_shape[0], :min_shape[1]] for fp in filepaths]
else:
    print("图像形状一致，无需调整。")
    arrays = [rasterio.open(fp).read(1) for fp in filepaths]

# 将多个单波段数组堆叠成一个多波段数组
stacked = np.stack(arrays)

# 创建一个掩膜，标记出 NoData 值的位置
nodata_mask = np.any(stacked == -9999, axis=0)  # 假设你的 NoData 值是 -9999

# 将多波段数组重塑为二维数组，以便进行 PCA 和 K-means
reshaped = stacked.reshape(stacked.shape[0], -1).T

# 只保留非 NoData 的值
valid_data = reshaped[~nodata_mask.ravel()]

# 标准化数据
scaler = StandardScaler()
scaled = scaler.fit_transform(valid_data)

# 进行 PCA
pca = PCA(n_components=2)  # 你可以根据需要调整主成分的数量
pca_result = pca.fit_transform(scaled)

# 进行 K-means 分类
kmeans = KMeans(n_clusters=2, n_init=10, random_state=0, init='k-means++')  # 使用 K-means++ 初始化方法，并设置随机种子
kmeans_result = kmeans.fit_predict(pca_result)

# 计算每个类别的平均值
class_means = [np.mean(pca_result[kmeans_result == i]) for i in range(2)]

# 根据平均值对类别标签进行排序，得到一个映射关系
sorted_labels = np.argsort(class_means)

# 定义等级标签
labels = ["低", "高"]

# 将分类结果映射到等级标签
classification_labels = np.array(labels)[sorted_labels][kmeans_result]

# 创建一个映射字典
label_dict = {"低": 1, "高": 2}

# 将等级标签转换为整数
classification_int = np.vectorize(label_dict.get)(classification_labels)

# 创建一个全是 NoData 值的数组，用于存储分类结果
classification = np.full_like(nodata_mask, -9999, dtype=np.int16)  # 假设你的 NoData 值是 -9999

# 将分类结果放入非 NoData 的位置
classification[~nodata_mask] = classification_int

# 保存分类结果为 GeoTIFF 文件
output_folder = "D:\\thesis\\新建文件夹 (3)"  # 你的输出文件夹
output_filepath = output_folder + "\\classification21.tif"  # 输出文件的完整路径

with rasterio.open(filepaths[0]) as src:
    profile = src.profile
    profile.update(dtype=rasterio.int16, count=1, compress='lzw', nodata=-9999)

with rasterio.open(output_filepath, 'w', **profile) as dst:
    dst.write(classification.astype(rasterio.int16), 1)
