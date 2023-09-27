# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 18:40
# @Author  : Sigrid
# @FileName: dataanalyze.py
# @Software: PyCharm

import pandas as pd
import matplotlib.pyplot as plt
import os

path = "data/"
# 修改filename 需要带后缀的
filename = 'windows_mining.xlsx'
# 读取文件到DataFrame
if filename.endswith('.xlsx'):
    df = pd.read_excel(path + filename)
elif filename.endswith('.csv'):
    df = pd.read_csv(path + filename)

# 计算平均值
mean_values = df.mean()

# 计算中位数
median_values = df.median()

# 计算方差
variance_values = df.var()

# 计算标准差
std_values = df.std()

# 创建DataFrame来保存统计结果
stats_df = pd.DataFrame({'Mean': mean_values,
                         'Median': median_values,
                         'Variance': variance_values,
                         'Standard Deviation': std_values})

# 分析结果存入
outpath = os.path.splitext(filename)[0]
out_dir = "analyze_result/" + outpath
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 写入Excel文件
stats_df.to_excel(out_dir + '/' +'statistics'+ outpath+'.xlsx', index=True)

# 写入CSV文件
stats_df.to_csv(out_dir + '/' +'statistics'+ outpath+'.csv', index=True)

# 生成直方图

# 从数据框中提取相应的指标列数据
time_data = df['time']
cpu_data = df['cpu%']
mem_data = df['mem%']
down_data = df['down']

# 创建画布和子图
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# 绘制cpu%的直方图
axes[0].bar(time_data, cpu_data, alpha=0.5, color='blue')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('CPU%')
axes[0].set_title(outpath+'_CPU%Histogram')

# 绘制mem%的直方图
axes[1].bar(time_data, mem_data, alpha=0.5, color='green')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Mem%')
axes[1].set_title(outpath+'_Mem%Histogram')

# 绘制down的直方图
axes[2].bar(time_data, down_data, alpha=0.5, color='yellow')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Down')
axes[2].set_title(outpath+'_DownHistogram')

plt.tight_layout()
plt.title(outpath+'Histogram')
plt.savefig(out_dir + '/'+outpath+'histogram.png')
plt.show()

# # 生成箱线图
# df[['cpu%', 'mem%', 'down']].plot(kind='box')
# plt.ylabel('Value')

# 创建画布和子图
fig2, axes = plt.subplots(3, 1, figsize=(8, 12))

# 绘制 CPU% 的箱线图
axes[0].boxplot(cpu_data)
axes[0].set_xticklabels(['CPU%'])
axes[0].set_ylabel('Value')
axes[0].set_title(outpath+'_CPU% Boxplot')

# 绘制 Mem% 的箱线图
axes[1].boxplot(mem_data)
axes[1].set_xticklabels(['Mem%'])
axes[1].set_ylabel('Value')
axes[1].set_title(outpath+'_Mem% Boxplot')

# 绘制 Down 的箱线图
axes[2].boxplot(down_data)
axes[2].set_xticklabels(['Down'])
axes[2].set_ylabel('Value')
axes[2].set_title(outpath+'_Down Boxplot')

plt.savefig(out_dir + '/' +outpath+ 'boxplot.png')
plt.show()

# 绘制对应的折线图

plt.plot(time_data, cpu_data, marker='o', label='CPU%', linewidth=1, markersize=3)
plt.plot(time_data, mem_data, marker='o', label='Mem%', linewidth=1, markersize=3)
plt.plot(time_data, down_data, marker='o', label='Down', linewidth=1, markersize=3)

plt.xlabel('Time')
plt.ylabel('Value')
plt.title(outpath+'Metrics over Time')
plt.legend()

plt.savefig(out_dir + '/' +outpath+ 'metrics_line_plot.png')
plt.show()

