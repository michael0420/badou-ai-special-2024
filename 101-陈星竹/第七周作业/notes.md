## 层次聚类

### linkage函数

````python
scipy.cluster.hierarchy.linkage(y, method='single', metric='euclidean', optimal_ordering=False)

````

- **`linkage` 函数**是 `scipy.cluster.hierarchy`模块中的一个函数，用于计算层次聚类。它的参数包括：

  - `y`: 输入的距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）
  - `'method'`: 用于计算距离的方法，常见的方法包括：
    - `'single'`: 最近点法
    - `'complete'`: 最远点法
    - `'average'`: 均值法
    - `'ward'`: Ward 最小方差法（默认）
    - 其他方法还有 `'centroid'`， `'median'` 等。

  * `metric`: 距离度量方法，当输入数据是二维数组时使用。常用的距离度量方法包括 `'euclidean'`（欧几里得距离）、 `'cityblock'`（曼哈顿距离）等。默认是 `'euclidean'`。 

  * `optimal_ordering`: 布尔值，如果为 True，将重新排列聚类顺序，使得树状图更具解释性。默认为 False。

  `linkage`函数返回一个矩阵 `Z`,这个矩阵包含了执行层次聚类所需的合并步骤信息。一个 (𝑛−1)×4(*n*−1)×4 的二维数组，其中每一行表示一个合并步骤。每行包含以下四个值：

  - `Z[i, 0]` 和 `Z[i, 1]`: 被合并的两个簇的索引。
  - `Z[i, 2]`: 这两个簇之间的距离。
  - `Z[i, 3]`: 新簇中的样本数。

### 聚类分配

```python
scipy.cluster.hierarchy.fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None)

```

- **`fcluster` 函数**:  也是 `scipy.cluster.hierarchy` 模块中的函数，用于将层次聚类的结果切分成具体的簇。它的参数包括：

  - `Z`: 由 `linkage` 函数生成的层次聚类树矩阵。
  - `t`: 切分簇的临界值。
  - `criterion`: 切分簇的依据。常见的选项包括：
    - `'inconsistent'`: 根据不一致系数切分（默认）。
    - `'distance'`: 根据最大聚类距离切分。
    - `'maxclust'`: 根据最大簇数切分。
    - 其他选项还有 `'monocrit'` 等。
  - `depth`: 当 `criterion` 为 `'inconsistent'` 时使用，表示计算不一致系数时的深度。
  - `R`: 预先计算的不一致矩阵，可以省略。
  - `monocrit`: 当 `criterion` 为 `'monocrit'` 时使用的单一临界值。

  `fcluster` 返回一个数组 `f`，一维数组，表示每个样本所属的簇的标签

### 绘制聚类树（树状图）

```python
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
```

- **`plt.figure` 函数**: 创建一个新的图形对象，用于绘制。`figsize` 参数指定图形的尺寸（宽度为5英寸，高度为3英寸）。
- **`dendrogram` 函数**: 绘制聚类树。参数 `Z` 是由 `linkage` 函数生成的层次聚类树矩阵。这个函数返回一个字典，包含了用于绘制聚类树的各种信息。

### 打印层次聚类矩阵并显示图形

```python
print(Z)
plt.show()
```

- **`print(Z)`**: 打印由 `linkage` 函数生成的层次聚类树矩阵 `Z`。
- **`plt.show()`**: 显示绘制的图形。

### 层次聚类优缺点

#### 优点

* 距离和规则的相似度容易定义，限制少
* 不需要预先制定聚类数
* 可以发现类的层次关系

#### 缺点

* 计算复杂
* 算法可能会聚类成链状

## 密度聚类

#### 1. 介绍

密度聚类（Density-Based Spatial Clustering of Applications with Noise, DBSCAN）是一种基于密度的聚类算法。它可以识别数据中的高密度区域，并将这些区域中的数据点划分为一个个簇，同时能够识别噪声点。与基于距离的聚类方法（如 K-means）不同，DBSCAN 不需要预先指定簇的数量，并且能够处理形状任意的簇和噪声点。

#### 2. 算法参数

- **eps (ε)**: 两个样本点被视为邻居的最大距离。
- **min_samples**: 一个点要成为核心点所需要的最小邻居数（包括点本身）。

#### 3. 算法步骤

1. **核心点**: 如果一个点的邻居数（在距离 eps 内）大于或等于 min_samples，这个点被标记为核心点。
2. **边界点**: 如果一个点的邻居数小于 min_samples，但它位于某个核心点的邻域内，这个点是边界点。
3. **噪声点**: 既不是核心点，也不是边界点的点是噪声点。

#### 4. 应用场景

DBSCAN 非常适用于具有以下特征的数据集：

- 存在密集的簇和稀疏的噪声点。
- 簇的形状复杂，不规则。

#### 5. 使用步骤

下面是一段使用 DBSCAN 对 Iris 数据集进行聚类的代码示例，并对结果进行可视化：

```python
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn import datasets 
from sklearn.cluster import DBSCAN

# 加载数据集
iris = datasets.load_iris() 
X = iris.data[:, :4]  # 取特征空间中的4个维度

# 使用DBSCAN进行聚类
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X) 
label_pred = dbscan.labels_

# 检查标签
unique_labels = np.unique(label_pred)
print("Unique labels:", unique_labels)

# 根据聚类标签分离数据点并绘制
colors = ['red', 'green', 'blue', 'purple', 'yellow']
markers = ['o', '*', '+', 's', 'd']

for label in unique_labels:
    if label == -1:
        # 绘制噪声点
        plt.scatter(X[label_pred == label][:, 0], X[label_pred == label][:, 1], c='black', marker='x', label='noise')
    else:
        plt.scatter(X[label_pred == label][:, 0], X[label_pred == label][:, 1], c=colors[label % len(colors)], marker=markers[label % len(markers)], label=f'label{label}')

# 设置轴标签和图例
plt.xlabel('sepal length')  
plt.ylabel('sepal width')  
plt.legend(loc=2)  # 设置图例在左上角

# 显示图像
plt.show()
```

#### 6. 代码解释

- **数据准备**: 加载 Iris 数据集，并取其前四个特征。
- **DBSCAN 聚类**: 设置 `eps` 和 `min_samples` 参数，并对数据集进行聚类，得到每个点的标签。
- **结果分析**: 使用 `np.unique` 查看聚类结果中的唯一标签，包括噪声点（标签为 -1）。
- **数据可视化**:
  - 使用 `plt.scatter` 函数绘制不同标签的点。
  - 不同标签用不同颜色和形状表示。
  - 噪声点用黑色 `x` 表示。
  - 设置图例的位置在左上角。
- **显示图像**: 调用 `plt.show()` 展示最终的散点图。

#### 7. 结果解释

- **标签为 0, 1, 2**: 表示聚类后的三个簇，每个簇用不同的颜色和形状表示。
- **噪声点**: 用黑色 `x` 表示，代表未归属于任何簇的点。
- **图例**: 用于区分不同的簇和噪声点，位于图的左上角。

#### 8. 优缺点

**优点**:

- 不需要预先指定簇的数量。
- 能够识别任意形状的簇。
- 对噪声有很好的鲁棒性。

**缺点**:

- 在高维数据中效果不佳（“维度灾难”问题）。
- 对参数 eps 和 min_samples 比较敏感，参数选择需要结合具体数据进行调整。

#### 9. 调整参数

- **eps**: 如果 eps 过小，可能会导致大部分点被标记为噪声；如果 eps 过大，不同簇可能会被合并。
- **min_samples**: 较大的 min_samples 值可以减少噪声点的数量，但可能会导致更少的簇。

通过调整参数 `eps` 和 `min_samples`，可以优化 DBSCAN 聚类的效果，适应不同的数据集特点。对于实际应用，需要进行实验和参数调优，以找到最佳的聚类效果。

## SIFT—尺度不变特征变换

#### 1. 简介

SIFT（Scale-Invariant Feature Transform）是一种用于检测和描述局部特征的计算机视觉算法。由 David Lowe 在1999年提出，并在2004年进行了进一步发展。SIFT 特征对缩放、旋转以及部分仿射变换具有不变性，并对光照变化具有一定的鲁棒性，因此广泛应用于图像匹配、物体识别、三维重建等领域。

#### 2. SIFT 的主要步骤

1. **尺度空间极值检测（Scale-space Extrema Detection）**：
   - 使用高斯核在不同尺度下对图像进行平滑，并计算高斯差分（Difference of Gaussian, DoG）。
   - 在不同尺度的图像中检测极值点，找到潜在的关键点。
2. **关键点定位（Keypoint Localization）**：
   - 通过拟合精确位置和尺度，过滤掉对比度低的关键点和边缘效应较强的关键点。
3. **方向分配（Orientation Assignment）**：
   - 在关键点的邻域内计算梯度方向直方图，并根据最大峰值的方向分配主方向，使得描述符对旋转具有不变性。
4. **关键点描述符（Keypoint Descriptor）**：
   - 在关键点的邻域内计算梯度的方向和幅值，生成描述符。SIFT 描述符通常是一个128维的向量。

#### 3. SIFT 实现步骤

下面的代码实现了使用 OpenCV 库的 SIFT 算法，对图像进行关键点检测和绘制。

```python
import cv2
import numpy as np

# 加载图像
img = cv2.imread("iphone1.png")

# 检查图像是否成功加载
if img is None:
    print("Error: Unable to load image. Check the file path.")
    exit()

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建 SIFT 对象
sift = cv2.SIFT_create()

# 检测关键点和计算描述符
keypoints, descriptor = sift.detectAndCompute(gray, None)

# 绘制关键点
img_with_keypoints = cv2.drawKeypoints(image=img, keypoints=keypoints,
                                       outImage=None,
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                       color=(51, 163, 236))

# 显示结果图像
cv2.imshow('sift_keypoints', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 4. 代码解释

1. **加载图像**:

   ```python
   img = cv2.imread("iphone1.png")
   ```

   通过 OpenCV 的 `cv2.imread` 函数加载图像。路径应根据实际情况进行调整。

2. **检查图像是否成功加载**:

   ```python
   if img is None:
       print("Error: Unable to load image. Check the file path.")
       exit()
   ```

   确保图像文件存在且路径正确。如果图像未加载成功，打印错误消息并退出程序。

3. **转换为灰度图像**:

   ```python
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   ```

   使用 `cv2.cvtColor` 函数将图像从 BGR 转换为灰度图像，这是 SIFT 算法的要求。

4. **创建 SIFT 对象**:

   ```python
   sift = cv2.SIFT_create()
   ```

   创建一个 SIFT 对象，用于检测关键点和计算描述符。

5. **检测关键点和计算描述符**:

   ```python
   keypoints, descriptor = sift.detectAndCompute(gray, None)
   ```

   在灰度图像中检测关键点，并计算每个关键点的描述符。`keypoints` 是一个包含所有关键点的列表，`descriptor` 是一个包含每个关键点描述符的矩阵。

6. **绘制关键点**:

   ```python
   img_with_keypoints = cv2.drawKeypoints(image=img, keypoints=keypoints,
                                          outImage=None,
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                          color=(51, 163, 236))
   ```

   使用 `cv2.drawKeypoints` 函数在原始图像上绘制关键点。`cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` 标志会绘制每个关键点的方向和大小。

7. **显示结果图像**:

   ```python
   cv2.imshow('sift_keypoints', img_with_keypoints)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```

   显示包含关键点的图像，并等待按键关闭窗口。

#### 5. SIFT 的优缺点

**优点**:

- **尺度不变性**: 对图像的缩放具有不变性。
- **旋转不变性**: 对图像的旋转具有不变性。
- **鲁棒性**: 对光照变化、噪声等具有较强的鲁棒性。
- **丰富的描述符**: 提供的描述符可以用于图像匹配、对象识别等多种任务。

**缺点**:

- **计算复杂度高**: 由于涉及多尺度高斯平滑和梯度计算，SIFT 算法计算复杂度较高。
- **专利问题**: 原始的 SIFT 算法受专利保护，可能限制了其在某些商业应用中的使用（不过现在已过专利期）。

#### 6. 总结

SIFT 是一种经典的特征检测和描述算法，具有尺度不变性、旋转不变性和较强的鲁棒性。尽管计算复杂度较高，但其优越的性能使其在计算机视觉领域得到广泛应用。通过 OpenCV，可以方便地使用 SIFT 进行图像关键点检测和描述符计算。以上代码展示了如何加载图像、检测关键点并绘制结果图像，是学习和应用 SIFT 的一个很好的起点。
