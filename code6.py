import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用 'Agg' 后端

# 设置matplotlib的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果你的电脑上没有'SimHei'字体，你需要选择一个支持中文的字体
plt.rcParams['axes.unicode_minus'] = False

# 读取Excel数据
df = pd.read_excel("D:/thesis/工作簿2013_.xlsx", index_col=0)

# 计算单一土地利用动态度
slud = df.pct_change(axis=1).dropna(axis=1, how='all') / 4 * 100

# 对数据进行归一化处理
slud_normalized = (slud - slud.min()) / (slud.max() - slud.min())

# 创建保存结果的目录（如果它们不存在）
os.makedirs("D:/thesis/SLUD", exist_ok=True)

# 将单一土地利用动态度保存到CSV文件中
slud_normalized.to_csv("D:/thesis/SLUD/单一土地利用动态度.csv")

# 绘制各个类型的折线图
plt.figure(figsize=(10, 6))
for land_type in slud_normalized.index:
    plt.plot(slud_normalized.columns, slud_normalized.loc[land_type], label=land_type)
plt.xlabel('年份')
plt.ylabel('单一土地利用动态度（归一化）')
plt.legend()
plt.grid(True)
plt.title('时间与单一土地利用动态度')
plt.savefig("D:/thesis/SLUD/单一土地利用动态度.png", dpi=300)  # 设置dpi为300
plt.show()
