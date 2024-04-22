import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel(r"D:\thesis\21land\merged_land.xlsx")

# 按照LID值的大小对行进行排序
df = df.sort_values('LID')

# 确保LID列包含从1到1646的所有值
for lid_value in range(1, 1647):
    if lid_value not in df['LID'].values:
        # 如果某个值缺失，添加一行，其中LID列的值为缺失的值，其余列的值为0
        missing_row = pd.DataFrame([[lid_value] + [0]*(len(df.columns)-1)], columns=df.columns)
        df = pd.concat([df, missing_row])

df = df.sort_values('LID')  # 重新排序以确保LID列的值是递增的

# 保存到新的Excel文件
df.to_excel(r"D:\thesis\21land\merged_land_sorted.xlsx", index=False)
