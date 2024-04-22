import pandas as pd
import os
import glob

def merge_csv(files_dir, outpath): # 将多个csv文件合并为一个，待合并csv集目录，合并后csv文件路径
    csv_files = glob.glob(os.path.join(files_dir, "*.csv")) # 抓取目录下全部csv文件
    features = pd.DataFrame() # 生成一个DF格式要素
    for csv_file in csv_files:
        feature = pd.read_csv(csv_file) # 读取单个csv文件
        features = features.append(feature) # 添加进要素集
    features.to_csv(outpath, index=False, encoding="utf-8") # 导出为csv文件
    print("merged successfully!")

def split_csv(csv_file, outdir, count): # 将一个csv文件拆分为多个，待拆分csv文件，拆分保存目录，拆分个数
    feature = pd.read_csv(csv_file) # 读取csv
    feature = pd.DataFrame(feature)
    num = int(len(feature)/count) # 单个文件个数
    for i in range(count):
        start = i * num
        end = (i+1) * num
        if i == 0:
            start = 0
        if i == count-1:
            end = len(feature)
        feature_single = feature.iloc[start:end, :]
        file_name = os.path.splitext(os.path.split(csv_file)[1])[0]
        comlete_name = f"{file_name}_{str(i+1)}_{end-start}count"
        outpath = os.path.join(outdir, comlete_name + ".csv")
        feature_single.to_csv(outpath, index=False, encoding="utf-8") # 导出为csv文件
        print(f"{comlete_name} split successfully")

model = input("请输入模式(1：合并 2：拆分)：")
if model == "1":
    print("\n==================== 进入合并模式 ====================\n")
    files_dir = input("请输入待合并的csv文件集目录：")
    outpath = input("请输入合并后文件的输出路径：")
    merge_csv(files_dir, outpath)
    print("\n=====================================================")
if model == "2":
    print("\n==================== 进入拆分模式 ====================\n")
    csv_file = input("请输入待拆分的csv文件路径：")
    outdir = input("请输入拆分后文件集的输出路径：")
    counts = input("请输入该csv文件的拆分个数：")
    counts = int(counts)
    split_csv(csv_file, outdir, counts)
    print("\n=====================================================")
