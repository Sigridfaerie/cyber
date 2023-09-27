# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 16:18
# @Author  : Sigrid
# @FileName: train_cnn.py
# @Software: PyCharm
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataLoader import dataLoader
from cnn_model import CNNModel
import tensorflow as tf
from sklearn.model_selection import train_test_split

#filename 填写的是文件的名字 不需要加后缀 .csv
filename = 'windows_data'

# step 1 ：加载dataLoader
normalized_features, one_hot_labels, train_data = dataLoader(filename + '.csv')

# Step 2: 划分数据集
X_train, X_val, y_train, y_val = train_test_split(normalized_features, one_hot_labels, test_size=0.2, random_state=42)

# Step 3: 创建模型实例
num_classes = len(np.unique(y_train))
model = CNNModel(num_classes)

# Step 4: 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 5: 训练模型
batch_size = 32
epochs = 10

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

# 训练的指标存入txt
out_dir = "output/" + filename
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(out_dir + '/' + 'training_epochs.txt', 'w') as f:
    f.write("Loss\tAccuracy\n")
    for loss, accuracy in zip(history.history['loss'], history.history['accuracy']):
        f.write(f"{loss}\t{accuracy}\n")

# 绘制训练过程中的损失和准确率变化图
plt.figure(figsize=(12, 6))
epochs_range = range(1, epochs+1)

# 绘制损失变化曲线
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['loss'], 'b-', label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], 'r-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# 绘制准确率变化曲线
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['accuracy'], 'b-', label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], 'r-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# 保存训练过程中的损失和准确率变化图
plt.savefig(out_dir + '/' + 'training_metrics.png')
plt.show()

# 在测试集上进行预测
predictions = model(X_val, training=False)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_val, axis=1)

# 保存以上的产出
model.save(out_dir + '/'+'trained_model_'+ filename)

# 计算混淆矩阵
confusion_matrix = tf.math.confusion_matrix(true_labels, predicted_labels)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(out_dir + '/' + 'confusion_matrix.png')
plt.show()

