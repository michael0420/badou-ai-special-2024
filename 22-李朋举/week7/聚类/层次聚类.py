###cluster.py
#导入相应的包
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

'''
linkage函数执行的层次聚类分析是一种基于距离的聚类方法，通过计算数据点之间的距离或相似性来将它们分组。
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
   若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
   若y是2维观测向量，则在层次聚类中，特征矩阵通常表示数据点之间的距离或相似性。
   
2. method是指计算类间距离的方法。
    method=’single’：指定聚类算法。在这里，method=’single’表示使用单链接（Single Linkage）算法进行聚类。
                     单链接算法是一种基于距离的聚类方法，它在每次合并两个簇时，选择两个簇中距离最近的两个数据点作为新簇的种子。
    method=’ward’:   是一个距离度量标准，它表示使用沃德距离（Ward's distance）作为聚类的依据。
    沃德距离是一种基于方差的距离度量标准，它试图最小化每个簇内的方差，并最大化不同簇之间的方差。这种方法在层次聚类分析中常被使用，因为它可以有效地发现自然的聚类结构。
    
3.metric=’euclidean’：指定距离度量标准。在这里，metric=’euclidean’表示使用欧氏距离（Euclidean distance）作为距离度量标准。
                      欧氏距离是一种常见的距离度量标准，它在二维空间中表示两个数据点之间的直线距离。
    返回一个Z矩阵，该矩阵表示数据点之间的聚类关系。Z矩阵的元素是整数，表示数据点所属的簇。例如，如果Z[i, j]的值为 1，则表示数据点i和j属于同一个簇；如果Z[i, j]的值为 0，则表示数据点i和j不属于同一个簇。
'''


'''
fcluster 是 scipy.cluster.hierarchy 模块中的一个函数，用于从层次聚类中形成的链接矩阵（通常是通过 linkage 函数生成的）中提取聚类
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
    Z: array_like
        描述: 链接矩阵，通常是通过 linkage 函数生成的。这个矩阵编码了层次聚类过程中各个簇的合并顺序。
        形状: (n-1, 4)，其中 n 是原始数据点的数量。
        内容: 每一行代表一次合并操作，包含四个元素：[i, j, u, v]，其中 i 和 j 是被合并的簇的索引，u 和 v 是这两个簇在合并后的新簇中的索引。
    t: float
        描述: 截断距离。所有在链接矩阵中距离小于或等于 t 的簇都将被合并到同一个簇中。
        用法: 根据特定的 criterion 参数（见下文），选择适合的 t 值可以产生不同粒度的聚类结果。
        criterion: str, optional
        描述: 用于确定哪些簇应该被合并的标准。
        可选值:
        'inconsistent': 使用不一致系数作为合并标准。这是默认的选项。
        'distance': 使用原始的距离作为合并标准。
        'maxclust': 合并簇直到达到或超过 t 值的簇的数量。  √ 
        用法: 根据你的聚类需求选择合适的标准。例如，'inconsistent' 标准通常会产生更紧凑的簇，而 'distance' 标准可能会产生更松散的簇。
    depth: int, optional
        描述: 当 criterion 为 'maxclust' 时，这个参数指定了最大簇的数量。
        用法: 例如，depth=2 表示将数据集分割成最多2个簇。
    R: array_like, optional
        描述: 用于不一致系数的阈值矩阵。通常不需要手动设置，除非你有特定的需求。
        形状: (n, n)，其中 n 是原始数据点的数量。
        用法: 如果提供了 R，则 fcluster 会使用 R 而不是从 Z 中计算不一致系数。
    monocrit: str, optional
        描述: 用于 'distance' 标准的单标准类型。当 criterion 为 'distance' 时，此参数可用于选择如何计算距离。
        可选值: 可以是 'median' 或 'mean'。
        用法: 例如，monocrit='median' 表示使用簇中所有点对的距离的中位数作为簇间距离。
        返回值:
    fcluster: ndarray
        形状: (n,)
        内容: 一个整数数组，其中每个整数表示对应数据点所属的簇的索引。
        注意：在使用 fcluster 之前，通常需要先使用 linkage 函数从原始数据生成链接矩阵 Z。
'''

X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
Z = linkage(X, 'ward')
f = fcluster(Z,4,'distance')

fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
'''
   第 0 次 ：   [[0.         3.         0.         2.        ]
   第 1 次 ：    [4.         5.         1.15470054 3.        ]
   第 2 次 ：    [1.         2.         2.23606798 2.        ]
   第 3 次 ：    [6.         7.         4.00832467 5.        ]]
'''
plt.show()


'''
plt.figure()：这是一个用于创建一个新的图形窗口的函数。
    figsize=(5, 3)：这是figure函数的一个参数，用于设置图形的尺寸。
         它是一个元组，第一个元素是图形的宽度（以英寸为单位），
                    第二个元素是图形的高度（以英寸为单位）。在这个例子中，图形的宽度被设置为5英寸，高度被设置为3英寸。
    fig：这是一个变量，用于存储plt.figure()返回的图形对象。这个对象可以用于进一步定制图形，例如添加子图、设置标题、调整边距等。
    在你创建了图形对象fig之后，你可以使用它来添加各种图形元素，例如通过fig.add_subplot()添加子图，或者通过fig.v()设置整个图形的标题。
    

dendrogram(Z) 是 Python 中 scipy.cluster.hierarchy 模块中的一个函数，用于根据链接矩阵 Z（通常是通过层次聚类算法如 linkage 函数生成的）绘制树状图（dendrogram）。
    树状图是一种可视化工具，用于展示层次聚类中各个簇之间的合并顺序和相对距离。它可以帮助用户理解数据的聚类结构，以及聚类过程中簇是如何逐步形成的。
    dendrogram 函数返回一个对象，通常被赋值给变量 dn，这个对象包含有关树状图的信息，例如叶子节点的位置、颜色和标签等。这些信息可以用于进一步定制树状图的外观。
    下面是 dendrogram 函数的一些常见参数及其描述：
    Z: 链接矩阵，通常通过 linkage 函数生成。
    orientation: 树状图的方向，可以是 'top'、'bottom'、'left' 或 'right'。
    labels: 数据点的标签，用于在树状图的叶子节点上显示。
    distance_sort: 是否根据距离对树状图的叶子节点进行排序。
    show_leaf_counts: 是否在树状图的叶子节点旁显示数据点的数量。
    leaf_rotation: 叶子节点的旋转角度。
    leaf_font_size: 叶子节点标签的字体大小。
    使用 dendrogram 函数后，通常还需要调用 plt.show()（如果你已经导入了 matplotlib.pyplot 作为 plt）来显示树状图。
'''