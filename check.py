import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('D:\\thesis\\Markov\\prediction_areas.xlsx')

# 转置 DataFrame
df_transposed = df.transpose()

# 将转置后的 DataFrame 保存为新的 Excel 文件
df_transposed.to_excel('D:\\thesis\\Markov\\prediction_areas_transposed.xlsx')
