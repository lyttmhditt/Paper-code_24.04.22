import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置字体为SimHei以显示中文字符
fontP = font_manager.FontProperties()
fontP.set_family('SimHei')
fontP.set_size(14)  # 设置字体大小
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 文件路径
file_path = "D:\\thesis\\240421\\7.xlsx"

# 读取Excel文件
df = pd.read_excel(file_path)

# 更改列名为中文名称
df.columns = ['年份', '水域', '城市', '耕地', '草地', '裸地']

# 设置图形大小
plt.figure(figsize=(10, 6))

# 定义颜色列表
colors = ['blue', 'green', 'red', 'purple', 'orange']

# 绘制折线图
for i, column in enumerate(df.columns[1:]):
    plt.plot(df[df.columns[0]], df[column], marker='o', label=column, color=colors[i])

# 显示图例
plt.legend(prop=fontP)

# 指定保存图形的路径
output_path = "D:\\thesis\\240421\\figure7.png"

# 保存图形
plt.savefig(output_path)

# 显示图形
plt.show()
