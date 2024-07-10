import tensorflow as tf
import tensorflow.keras as keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

"""实现鸢尾花分类
    数据集有三个分类，使用数字编码
"""
if __name__ == '__main__':
    # tensorflow v1版本立即执行
    tf.compat.v1.enable_eager_execution()
    # 加载数据集
    iris = load_iris()
    data, labels, labels_names = iris.data, iris.target, iris.target_names
    # print(labels_names)
    # exit(0)
    # 训练轮数，训练轮数根据数据集大小进行确定，便于模型收敛
    epochs = 1000
    # 将数据集按照8-2原则分割为训练集和测试集
    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=.2, random_state=0)
    # 使用keras构建神经网络
    net = keras.Sequential()
    net.add(keras.layers.Dense(3, input_shape=(4,)))
    net.add(keras.layers.Dense(20, activation=keras.activations.relu))
    net.add(keras.layers.Dense(3, activation=keras.activations.softmax))
    # 鸢尾花有三种分类属于多分类问题
    loss = keras.losses.SparseCategoricalCrossentropy()
    # 编译神经网络
    net.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    # 将训练集喂给神经网络进行训练
    net.fit(train_data, train_label, epochs=epochs)
    # 训练完成后使用测试集测试模型性能
    loss, accuracy = net.evaluate(test_data, test_label)
    print("loss: ", loss)
    print("accuracy: ", accuracy)
    # 取原始数据集中第一个数据进行预测
    result = net.predict(tf.expand_dims(data[0], axis=0))
    # 各个分类的概率
    print("the probability: ", result)
    index = np.argmax(result)
    # 选取最大概率对应的分类名称
    print("the category is: ", labels_names[index])
