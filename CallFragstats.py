# coding=utf8
import os
import time
from glob import glob
import pandas as pd


def frg_cmd(model, scripts_dir, output_dir): # 模型路径，脚本所在目录，输出目录
    csv_scripts = glob(os.path.join(scripts_dir, "*.csv")) # 抓取目录下全部csv文件
    for csv_script in csv_scripts:
        output_file = os.path.join(output_dir + ".class")
        print("=====================================================")
        print(f"model: {model}")
        print(f"script: {csv_script}")
        print(f"output: {output_dir}")
        print("\n")
        os.system(f"frg -m {model} -b {csv_script} -o {output_file}") # 命令行执行
        read_csv_script = pd.read_csv(csv_script) # 读取脚本文件
        read_output_file = pd.read_csv(output_file) # 读取输出文件
        read_csv_script = pd.DataFrame(read_csv_script)
        read_output_file = pd.DataFrame(read_output_file)

        print("\n")
        print(f"{output_file} is completed!")
        print("=====================================================")

# model = input("请输入模型路径：")
# scripts_dir = input("请输入脚本目录：")
# output_dir = input("请输入输出目录：")
model = r"D:\\Homework\\Endterm\\fragproject\\projectall.fca"
scripts_dir = r"D:\\Homework\\Endterm\\Afterdivide21"
output_dir = r"D:\\Homework\\Endterm\\Result\\hebishi21"
frg_cmd(model, scripts_dir, output_dir)