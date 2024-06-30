'''
采用sklearn.cluster库Kmeans聚类分析运动员能力
'''
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
'''
第一列表示球员每分钟助攻数：assists_per_minute
第二列表示球员每分钟得分数：points_per_minute
'''
data = np.array([[0.0888, 0.5885],
                [0.1399, 0.8291],
                [0.0747, 0.4974],
                [0.0983, 0.5772],
                [0.1276, 0.5703],
                [0.1671, 0.5835],
                [0.1306, 0.5276],
                [0.1061, 0.5523],
                [0.2446, 0.4007],
                [0.1670, 0.4770],
                [0.2485, 0.4313],
                [0.1227, 0.4909],
                [0.1240, 0.5668],
                [0.1461, 0.5113],
                [0.2315, 0.3788],
                [0.0494, 0.5590],
                [0.1107, 0.4799],
                [0.1121, 0.5735],
                [0.1007, 0.6318],
                [0.2567, 0.4326],
                [0.1956, 0.4280]])

kmeans = KMeans(n_clusters=3)  # 声明函数，需求方一般可以提供，否则需要多试一些 K 值来保证更好的聚类效果
'''
KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')

n_clusters：即 K 值，可以随机设置，选择效果最好的；
init：即初始值选择的方式，默认是采用优化过的 k-means++ 方式，也可以自己指定中心点，或者采用 random 完全随机的方式。自己设置中心点一般是对于个性化的数据进行设置，很少采用。random 的方式则是完全随机的方式，一般推荐采用优化过的 k-means++ 方式；
n_init：初始化中心点的运算次数，默认 10。程序是否能快速收敛和中心点的选择关系非常大，所以在中心点选择上多花一些时间，来争取整体时间上的快速收敛还是非常值得的。由于每一次中心点都是随机生成的，这样得到的结果就有好有坏，非常不确定，所以要运行 n_init 次, 取其中最好的作为初始的中心点。如果 K 值比较大的时候，可以适当增大 n_init 这个值；
max_iter：最大迭代次数，如果聚类很难收敛的话，设置最大迭代次数可以及时得到反馈结果； 
algorithm：k-means 的实现算法，有“auto”、“full”、“elkan”三种，建议用默认的"auto"。"full"采用传统的 K-Means 算法，“auto”根据数据特点自动选择“full”还是“elkan”。
'''
result1 = kmeans.fit_predict(data)  # 传入数据，返回每个样本对应的簇的索引。Compute cluster centers and predict cluster index for each sample
'''
kmeans.fit -> kmeans.predict  = kmeans.fit_predict
'''
result2 = kmeans.fit(data)  # 拟合模型，计算Kmeans的聚类结果，族编号可能随机变化
result2 = result2.predict(data)  # 预测每个测试集X中的样本的所在簇，并返回每个样本所对应的簇的索引
print('一步聚类结果为：\n', result1)
print('分步聚类结果为：\n', result2)
'''
kmeans.fit -> result2.transform  = kmeans.fit_transform
'''
distance = kmeans.fit_transform(data)  # 簇距离空间，每个维度是样本点到集群中心的距离
print('样本点到集群中心的距离\n', distance)

x, y = data[:, 0], data[:, 1]
plt.xlabel('assistants per minute'), plt.ylabel('points per minute'), plt.title('Athlete Analysis')
# scatter绘制散点图，title图形名称，xlabel&ylabel横纵坐标轴名称，legend图例
scatter_A = plt.scatter(x[result1 == 0], y[result1 == 0], s=50, color='red', marker='s')
scatter_B = plt.scatter(x[result1 == 1], y[result1 == 1], s=50, color='green', marker='p')
scatter_C = plt.scatter(x[result1 == 2], y[result1 == 2], s=50, color='blue', marker='*')
'''
plt.scatter(x, y, s=150, c=result1, marker='*', alpha=0.7)
s表示形状大小， alpha表示透明度， label表示点的图例标签
marker默认表示为 'o' ,还有 'x' , '>' , '<', 'v' , '^' , '+' ,'s'方形, '*'五角星
'''
plt.legend(['scatter_A', 'scatter_B', 'scatter_C'], loc='upper left')
# plt.legend(handle=['scatter_A', 'scatter_B', 'scatter_C'], loc='upper right')
plt.show()

# # c是RGB或RGBA二维行数组，表示的是色彩序列，默认蓝色’b’。c不是单一RGB数字，也不是RGBA序列，因为不便区分。
# # c=result1聚类结果是0， 1， 2， 编号最大到黄色，最小到黑色，中间再均分
# data1 = [i for i in range(100)]
# data2 = [j for j in range(100)]
# plt.scatter(data1, data2, c=data2, marker='p')
# plt.colorbar()
# plt.show()
