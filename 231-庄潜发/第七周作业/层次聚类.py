"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/5/25 9:09
"""
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。
    method = ‘single’ 最邻近算法  min()
    method = 'complete' 最近邻点算法。 max()
    method = 'average' UPGMA算法（非加权组平均）法
    method = 'weighted' WPGMA（加权分组平均）法
    method = 'centroid' 
    method = 'median'
    method = 'ward' 
'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''
y = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
# 计算距离矩阵, 得到层次聚类的层次信息
Z = linkage(y, method='ward')
# [[0.         3.         0.         2.        ]
#  [4.         5.         1.15470054 3.        ]
#  [1.         2.         2.23606798 2.        ]
#  [6.         7.         4.00832467 5.        ]]
# 第一个参数和第二个参数是比较数据的序号,序号0和序号3的数据进行比较
# 第三个参数是类之间的距离,序号0和序号3它们数据一致,所以距离为0, 会进行第一次合并得到序号0和序号3的类,生成类别5
# 第四个参数是类里面有多少数据, 如果类的数据等于我们输入的样本数, 那么就达到终止条件

# 参数(linkage得到的矩阵, 4为迭代次数, distance显示距离)
f = fcluster(Z, 4, criterion='distance')
print(Z, f)
# 图像化界面
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)  # 打印表格
plt.show()
