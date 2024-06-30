import numpy as np
import matplotlib.pyplot as plt
import os

class K_Means():

    def __init__(self):
        self.num_clusters = None

    def fit(self, src_data, num_clusters, max_iter=10, tol=1e-10):
        # 设置聚类数量
        self.num_clusters = num_clusters

        # 初始化聚类中心
        centroids = np.array(src_data[:self.num_clusters], dtype=float)

        for i in range(max_iter):
            # 将样本分配到最近中心点的类中
            labels = np.zeros(src_data.shape[0]).astype("uint8")
            for j in range(src_data.shape[0]):
                distance = np.linalg.norm(centroids-src_data[j], ord=2, axis=1)
                labels[j] = distance.argmin()

            # 保存原来中心点
            pre_centroids = centroids.copy()

            # 更新中心点
            for j in range(self.num_clusters):
                cluster =  src_data[labels==j]
                if len(cluster) == 0:
                    continue
                centroids[j] = np.mean(cluster, axis=0)

            # 如果中心点无变化，则退出循环
            if np.all(np.linalg.norm(centroids-pre_centroids, ord=2, axis=1) < tol ):
                break

        return centroids, labels


def display_imgs(imgs, titles, rows, cols):
    for i in range( len(imgs) ):
        plt.subplot(rows, cols, i+1)
        plt.imshow(imgs[i])
        plt.title(titles[i])
        plt.xticks([]) , plt.yticks([]) # 不显示横纵坐标轴
    plt.show()


if __name__ == "__main__":
    # 设置图片路径
    base_dir = "img"
    img_name = "lenna.jpg"
    img_path = os.path.join(base_dir,img_name)

    # 读取图片
    img = plt.imread(img_path)

    # 将2维图片拉直为1维数据
    src_data = img.reshape(-1, img.shape[2])

    # 加载k_means算法
    my_k_means = K_Means()

    # 聚4类
    centroids_4, labels_4 = my_k_means.fit(src_data, 4)

    # 聚16类
    centroids_16, labels_16 = my_k_means.fit(src_data, 8)

    # 聚64类
    centroids_64, labels_64 = my_k_means.fit(src_data, 16)

    # 计算转化后的图像
    dst_img_4 = centroids_4[labels_4].astype("uint8").reshape(img.shape)
    dst_img_16 = centroids_16[labels_16].astype("uint8").reshape(img.shape)
    dst_img_64 = centroids_64[labels_64].astype("uint8").reshape(img.shape)

    # 展示图像
    imgs = [img, dst_img_4, dst_img_16, dst_img_64]
    titles = ["src_img", "dst_img_4", "dst_img_16", "dst_img_64"]
    display_imgs(imgs, titles, rows=2, cols=2)

