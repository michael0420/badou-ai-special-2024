# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


def kmeans_clustering_and_display(img, k):
    # 图像二维像素转换为一维
    data = img.reshape((-1, 3))
    data = np.float32(data)

    # 停止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # K-Means聚类
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)

    # 图像转换回uint8二维类型
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)

    # 图像转换为RGB显示
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    return segmented_image


# 读取原始图像（默认是BGR格式）
img = cv2.imread('lenna.png')
if img is None:
    print("Error: 图像无法读取。")
    exit()

# 将BGR图像转换为RGB图像
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 初始化图像列表
images = [img_rgb]
titles = [u'原始图像']


# 对不同的K值进行聚类并显示结果
for k in [2, 4, 8, 16, 64]:
    segmented_image = kmeans_clustering_and_display(img, k)
    images.append(segmented_image)
    titles.append(f'聚类图像 K={k}')

# 设置matplotlib参数
plt.rcParams['font.sans-serif'] = ['SimHei']

# 使用subplot显示图像
fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flatten()):
    ax.imshow(images[i])
    ax.set_title(titles[i])

plt.tight_layout()
plt.show()