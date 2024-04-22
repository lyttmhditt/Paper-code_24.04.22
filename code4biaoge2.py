import pandas as pd
import os

# 创建一个字典，包含了所有的数据
data = {
    '地名': ['淇县', '淇滨区', '浚县', '鹤山区', '山城区'],
    '区划码': ['410622', '410611', '410621', '410602', '410603'],
    '第一产业增加值（亿元）': [23.4, 72.52, 31.3, 36.46, 5.14],
    '第二产业增加值（亿元）': [151.6, 615.93, 159.2, 240.53, 84.59],
    '第三产业增加值（亿元）': [66.8, 376.19, 103.3, 181.52, 40.80],
}

# 使用pandas创建一个DataFrame
df = pd.DataFrame(data)

# 将数字字段转换为数值类型
num_cols = ['第一产业增加值（亿元）', '第二产业增加值（亿元）', '第三产业增加值（亿元）']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# 将非数字字段转换为文本类型
text_cols = ['地名', '区划码']
df[text_cols] = df[text_cols].astype(str)

# 定义文件路径
file_path = os.path.join("D:\\thesis", "cyzjz2.xlsx")

# 将DataFrame保存为Excel文件
df.to_excel(file_path, index=False, engine='openpyxl')
import pandas as pd
import os

# 创建一个字典，包含了所有的数据
data = {
    '地名': ['淇县', '淇滨区', '浚县', '鹤山区', '山城区'],
    '区划码': ['410622', '410611', '410621', '410602', '410603'],
    '第一产业增加值（亿元）': [23.4, 72.52, 31.3, 36.46, 5.14],
    '第二产业增加值（亿元）': [151.6, 615.93, 159.2, 240.53, 84.59],
    '第三产业增加值（亿元）': [66.8, 376.19, 103.3, 181.52, 40.80],
}

# 使用pandas创建一个DataFrame
df = pd.DataFrame(data)

# 将数字字段转换为数值类型
num_cols = ['第一产业增加值（亿元）', '第二产业增加值（亿元）', '第三产业增加值（亿元）']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# 将非数字字段转换为文本类型
text_cols = ['地名', '区划码']
df[text_cols] = df[text_cols].astype(str)

# 定义文件路径
file_path = os.path.join("D:\\thesis", "cyzjz3.xlsx")

# 将DataFrame保存为Excel文件
df.to_excel(file_path, index=False, engine='openpyxl')
