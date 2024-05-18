# -*- coding: utf-8 -*-
'''
@File    :   PCA.py
@Time    :   2024/04/18 23:31:41
@Author  :   廖红洋 
'''

import sklearn.decomposition as dp
from sklearn.datasets import load_iris 
# 查询了sklearn的常用数据集，只有鸢尾花这个比较简单，数据比较少，适合写作业，其他的不是太多就是类别差异很大，不适合PCA

x,y=load_iris(return_X_y=True) # 解构赋值，x为属性具体值，有四个维度，y为不同对象，共150
pca=dp.PCA(n_components=2) # 设置降维器输出为两维，即 任意维->两维
reduced_x=pca.fit_transform(x) #对x进行降维，此时，三种鸢尾花的四个属性被降维成两个属性，画图部分省略