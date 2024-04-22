import os
import shutil
import pandas as pd

# 定义源文件夹和目标文件夹
source_dir = "D:\\thesis\\resullt"
class_dir = "D:\\thesis\\21class"
land_dir = "D:\\thesis\\21land"

# 创建目标文件夹（如果它们不存在）
os.makedirs(class_dir, exist_ok=True)
os.makedirs(land_dir, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_dir):
    # 如果文件名包含".class"，则将其移动到class_dir
    if ".class" in filename:
        shutil.move(os.path.join(source_dir, filename), class_dir)
    # 如果文件名包含".land"，则将其移动到land_dir
    elif ".land" in filename:
        shutil.move(os.path.join(source_dir, filename), land_dir)

# 定义一个函数来合并一个文件夹中的所有文件
def merge_files(dir_path, output_filename):
    data_frames = []
    for i, filename in enumerate(os.listdir(dir_path)):
        if filename != output_filename:
            df = pd.read_csv(os.path.join(dir_path, filename), header=None, sep=',')
            df[0] = df[0].apply(lambda x: os.path.splitext(os.path.basename(x))[0])  # 使用apply函数来应用os.path.basename到df[0]的每个元素，并将'.tif'替换为''
            if i > 0:  # 对于除第一个文件外的其他文件，我们跳过标题行
                df = df[1:]
            data_frames.append(df)
    merged_df = pd.concat(data_frames)
    # 使用strip函数去除每个元素的前导和尾随空格
    merged_df = merged_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    merged_df.to_excel(os.path.join(dir_path, output_filename), index=False, header=False)

# 合并class_dir和land_dir中的所有文件
merge_files(class_dir, "merged_class.xlsx")
merge_files(land_dir, "merged_land.xlsx")
