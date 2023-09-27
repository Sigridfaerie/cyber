# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 15:25
# @Author  : Sigrid
# @FileName: datapreprocess.py
# @Software: PyCharm

"""
这个文件用来处理xlsx和新加入的csv文件，并把它们按照要求变成一个用于训练的csv文件
"""

import os
import pandas as pd

# 设置文件夹路径和标签映射
folder_path = 'data'
label_mapping = {'mining': 'mining', 'without': 'without'}

# 存储包含Windows文件名和Linux文件名的数据
windows_data = []
linux_data = []
linux_new_data = []
linux_new1_data = []
linux_new2_data = []

# 处理每个文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        # 提取文件名中的关键字
        keywords = [keyword for keyword in label_mapping if keyword in file_name.lower()]

        # 增加label列
        label = 0
        if len(keywords) > 0:
            label = label_mapping[keywords[0]]

        # 读取Excel文件
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path)

        # 增加label列并填充数据
        df['Label'] = label

        # 将数据分为Windows和Linux文件
        if 'windows' in file_name.lower():
            windows_data.append(df)
        elif 'linux' in file_name.lower():
            linux_data.append(df)
            linux_new1_data.append(df)
    if file_name.endswith('.csv'):
        # 提取文件名中的关键字
        keywords = [keyword for keyword in label_mapping if keyword in file_name.lower()]

        # 增加label列
        label = 0
        if len(keywords) > 0:
            label = label_mapping[keywords[0]]

        # 读取csv文件
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        # 增加label列并填充数据
        df['Label'] = label

        # 添加新的文件
        if '60%' in file_name.lower():
            linux_new2_data.append(df)
#linux_new1_data.append(linux_new2_data)
linux_new_data = linux_new1_data + linux_new2_data
# 合并Windows文件并保存为CSV
if len(windows_data) > 0:
    windows_merged_data = pd.concat(windows_data)
    windows_merged_data.to_csv('windows_data.csv', index=False)

# 合并Linux文件并保存为CSV
if len(linux_data) > 0:
    linux_merged_data = pd.concat(linux_data)
    linux_merged_data.to_csv('linux_data.csv', index=False)

# 合并新的linux训练的数据文件
if len(linux_new_data) > 0:
    linux_merged_data = pd.concat(linux_new_data)
    linux_merged_data.to_csv('linux_add_data_cpu60%.csv', index=False)
