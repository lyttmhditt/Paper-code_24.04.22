import pandas as pd

# 创建一个字典，包含了所有的数据
data = {
    "年份": ["2013年", "2017年", "2021年"],
    "全市生产总值（亿元）": [622.12, 832.59, 1064.64],
    "第一产业增加值（亿元）": [61.04, 61.10, 72.52],
    "第一产业占比（%）": [9.8, 7.3, 6.8],
    "第二产业增加值（亿元）": [446.26, 537.15, 615.93],
    "第二产业占比（%）": [71.7, 64.5, 57.9],
    "第三产业增加值（亿元）": [114.82, 234.34, 376.19],
    "第三产业占比（%）": [18.5, 28.2, 35.3]
}

# 将字典转换为DataFrame
df = pd.DataFrame(data)

# 转置DataFrame
df_transposed = df.transpose()

# 将转置后的DataFrame写入Excel文件
df_transposed.to_excel(r"D:\thesis\map.gdb\鹤壁市GDP数据_transposed.xlsx", header=False)
