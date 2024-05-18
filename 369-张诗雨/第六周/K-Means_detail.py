import cv2
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data, num_clusters):
        self._data = data
        self._num_clusters = num_clusters

    def train(self, max_iterations):
        centers = self.__centers_init(self._data, self._num_clusters)
        num_examples = self._data.shape[0]
        closest_center_ids = np.zeros((num_examples, 1), dtype=int)
        for i in range(max_iterations):
            print(i)
            closest_center_ids = self.__centers_find_closest(self._data, self._num_clusters, centers)
            centers = self.__centers_compute(self._data, self._num_clusters, closest_center_ids)
        labels = np.zeros((num_examples, 1))
        for example_index in range(num_examples):
            labels[example_index] = centers[closest_center_ids[example_index]]
        return labels, centers, closest_center_ids

    def __centers_init(self, data, num_clusters):
        num_examples = data.shape[0]
        random_ids = np.random.permutation(num_examples)
        centers = data[random_ids[:num_clusters], :]
        return centers

    def __centers_find_closest(self, data, num_clusters, centers):
        num_examples = data.shape[0]
        closest_center_ids = np.zeros((num_examples, 1), dtype=int)
        for example_index in range(num_examples):
            distance = np.zeros((num_clusters, 1))
            for center_index in range(num_clusters):
                diff = data[example_index, :] - centers[center_index, :]
                distance[center_index] = np.sum(diff ** 2)
            closest_center_ids[example_index] = np.argmin(distance)
        return closest_center_ids

    def __centers_compute(self, data, num_clusters, closest_center_ids):
        num_features = data.shape[1]
        centers = np.zeros((num_clusters, num_features))
        for center_index in range(num_clusters):
            example_ids = closest_center_ids == center_index
            centers[center_index] = np.mean(data[example_ids.flatten(), :], axis=0)
        return centers

if __name__=='__main__':
    # 读取原始图像灰度颜色
    img = cv2.imread('../lenna.png', 0)

    # 获取图像高度、宽度
    rows, cols = img.shape[:]

    # 图像二维像素转换为一维
    data = img.reshape((rows * cols, 1))
    data = np.float32(data)

    #K-Means聚类 聚集成4类
    max_iterations = 5
    num_clusters = 4
    kmeans = KMeans(data, num_clusters)
    labels, centers, closest_center_ids = kmeans.train(max_iterations)

    # 生成最终图像
    dst = labels.reshape((img.shape[0], img.shape[1]))

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 显示图像
    titles = [u'原始图像', u'聚类图像']
    images = [img, dst]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
