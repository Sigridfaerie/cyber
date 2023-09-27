# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 15:17
# @Author  : Sigrid
# @FileName: cnn_model.py
# @Software: PyCharm

import tensorflow as tf


class CNNModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.fc1 = tf.keras.layers.Dense(32, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc3 = tf.keras.layers.Dense(128, activation='relu')
        self.fc4 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.flatten(x)
        x = self.fc3(x)
        output = self.fc4(x)
        return output

