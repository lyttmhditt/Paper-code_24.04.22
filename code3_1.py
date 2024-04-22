import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel("D:\\thesis\\21class\\merged_class.xlsx")

# 获取所有唯一的TYPE值
types = df['TYPE'].unique()

# 对于每个TYPE值，选择对应的行并保存为一个新的Excel文件
for type_value in types:
    sub_df = df[df['TYPE'] == type_value]
    # 按照LID值的大小对行进行排序
    sub_df = sub_df.sort_values('LID')
    # 确保LID列包含从1到1646的所有值
    for lid_value in range(1, 1647):
        if lid_value not in sub_df['LID'].values:
            # 如果某个值缺失，添加一行，其中LID列的值为缺失的值，TYPE列的值为当前表格的TYPE值，其余列的值为0
            missing_row = pd.DataFrame([[lid_value, type_value] + [0]*(len(sub_df.columns)-2)], columns=sub_df.columns)
            sub_df = pd.concat([sub_df, missing_row])
    sub_df = sub_df.sort_values('LID')  # 重新排序以确保LID列的值是递增的
    sub_df.to_excel(f"D:\\thesis\\21class\\merged_class_{type_value}.xlsx", index=False)
