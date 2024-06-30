#!/usr/bin/env python
# encoding=gbk

# 导入所需的库
import matplotlib.pyplot as plt  # 导入绘图库matplotlib
import sklearn.decomposition as dp  # 导入sklearn库中的降维模块

# 导入数据集加载函数
# 注意：在较新版本的scikit-learn中，load_iris函数是直接从sklearn.datasets模块中导入的
# from sklearn.datasets.base import load_iris
from sklearn.datasets import load_iris

# 加载鸢尾花数据集，返回属性数据x和标签数据y
x, y = load_iris(return_X_y=True)   # 加载数据，x表示数据集中的属性数据，y表示数据标签
print('x表示数据集中的属性数据\n',x)
print('y表示数据标签\n',y)
'''
load_iris()：这是scikit-learn库中用于加载鸢尾花数据集的函数。
return_X_y=True：这个参数设置为True，表示返回的数据集会被拆分成两个部分，一个是属性数据(x)，另一个是标签数据(y)。
如果设置为False或不指定，默认会返回一个Bunch对象，包含数据集的各个部分，如属性、标签、描述等。
x, y = load_iris(return_X_y=True)：这一行代码将返回的属性数据赋值给变量x，标签数据赋值给变量y。所以x包含了鸢尾花数据集的属性信息，而y包含了对应的标签信息，用来表示鸢尾花的类别。

属性数据(x)，另一个是标签数据(y)这些都是什么？
在机器学习中，通常将数据集分为两部分：属性数据和标签数据。
属性数据 (X)：属性数据通常表示输入数据，它包含了描述样本特征的信息。在监督学习中，属性数据也称为特征。每个样本都由一组属性值组成，这些属性可以是数字、类别或者其他类型的数据。在加载鸢尾花数据集时，属性数据包含了四个特征：花萼长度、花萼宽度、花瓣长度和花瓣宽度。
标签数据 (y)：标签数据是用来表示样本所属类别或者目标值的数据。在监督学习中，标签数据是我们希望模型预测的结果或者类别。每个样本都有一个对应的标签值。在加载鸢尾花数据集时，标签数据表示了每个样本所属的鸢尾花的类别，通常用数字表示，比如0、1、2分别代表三个不同的鸢尾花品种。
在这个特定的代码示例中，x变量保存了鸢尾花数据集的属性信息，而y变量保存了对应的标签信息。

如果在 load_iris() 函数中将 return_X_y 参数设置为 False，或者不指定该参数，则函数将返回一个包含完整数据集的 Bunch 对象。这个 Bunch 对象通常包含以下属性：
data：包含属性数据的二维数组，每一行代表一个样本，每一列代表一个特征。
target：包含标签数据的一维数组，每个元素代表对应样本的类别标签。
target_names：一个包含目标类别名称的数组。
feature_names：一个包含属性特征名称的数组。
DESCR：数据集的描述信息。
通过返回 Bunch 对象，可以方便地访问数据集的各个部分，如属性数据、标签数据以及数据集的描述信息。
'''
# 初始化加载PCA算法对象，设置降维后的主成分数目为2
pca = dp.PCA(n_components=2)
print("pca\n", pca)

# 对原始数据进行降维，保存在reduced_x中
reduced_x = pca.fit_transform(x)
print("reduced_x:\n", reduced_x)

# 初始化用于保存降维后的数据点的列表
red_x, red_y = [], []  # 存储类别为0的数据点的降维结果
blue_x, blue_y = [], []  # 存储类别为1的数据点的降维结果
green_x, green_y = [], []  # 存储类别为2的数据点的降维结果

# 根据数据点的类别将降维后的数据点保存在不同的列表中
for i in range(len(reduced_x)):    # 按鸢尾花的类别将降维后的数据点保存在不同的表中
    if y[i] == 0:  # 如果类别为0，添加到红色点的列表中
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:  # 如果类别为1，添加到蓝色点的列表中
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:  # 如果类别为2，添加到绿色点的列表中
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

# 绘制散点图
plt.scatter(red_x, red_y, c='r', marker='x')  # 绘制红色点
plt.scatter(blue_x, blue_y, c='b', marker='D')  # 绘制蓝色点
plt.scatter(green_x, green_y, c='g', marker='.')  # 绘制绿色点
plt.show()  # 显示图形

'''
marker 是用来指定散点图中每个点的标记样式的参数。
常用的几种标记样式包括：
'o'：圆圈
's'：正方形
'+'：加号
'x'：叉号
'D'：菱形
'.'：点
在你的代码中，marker='x'表示红色点的标记样式是叉号，marker='D'表示蓝色点的标记样式是菱形，marker='.'表示绿色点的标记样式是点。
'''