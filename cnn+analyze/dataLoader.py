# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 16:05
# @Author  : Sigrid
# @FileName: dataLoader.py
# @Software: PyCharm

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

def dataLoader(path):
    # 读取CSV文件
    data = pd.read_csv(path)

    # 提取所需的列和标签
    features = data[['cpu%', 'mem%', 'down']]
    labels = data['Label']

    # 打印提取的特征和标签
    print(features)
    print(labels)

    # 归一化
    # 创建MinMaxScaler对象
    scaler = MinMaxScaler()

    # 对features进行归一化/标准化
    normalized_features = scaler.fit_transform(features)

    # one-hot
    # 创建LabelEncoder对象
    label_encoder = LabelEncoder()

    # 对labels进行编码
    encoded_labels = label_encoder.fit_transform(labels)

    # 将编码后的标签转换为独热编码
    one_hot_labels = np.eye(len(np.unique(encoded_labels)))[encoded_labels]

    # 将处理后的特征与标签组合成训练样本
    train_samples = list(zip(normalized_features, one_hot_labels))

    return normalized_features, one_hot_labels, train_samples

# dataLoader('linux_data.csv')