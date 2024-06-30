import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def kmeans(img, k):
    # 判断输入图像是否为彩色图像
    if len(img.shape) == 3 and img.shape[2] == 3:  # 彩色图像
        input_type = 'color'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    else:  # 灰度图像或者其他情况
        input_type = 'gray'

    # 获取图像高度、宽度
    rows, cols = img.shape[:2]

    # 图像二维像素转换为一维
    if input_type == 'color':
        data = img.reshape((-1, 3))
    else:
        data = img.reshape((rows * cols, 1))
    data = np.float32(data)

    # 停止条件 (type, max_iter, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS

    # K-Means聚类
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)

    # 生成最终图像
    if input_type == 'color':
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        dst = res.reshape((rows, cols, 3))
    else:
        dst = labels.reshape((rows, cols))

    return dst

if __name__ == '__main__':
    # 读取图像
    img_path = r'E:\pycharm_code\cv\Original_Data\0216.jpg'
    img = cv2.imread(img_path)

    # 确认图像被正确读取
    if img is None:
        print("Error: 图像读取失败，请检查路径")
        exit()

    # 执行kmeans函数，聚类数为4
    result_img = kmeans(img, 16)

    # 设置中文字体
    font = FontProperties(fname=r'C:\Windows\Fonts\SimHei.ttf', size=14)

    # 显示原始图像和聚类图像
    titles = [u'原始图像', u'聚类图像']
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), result_img]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i], fontproperties=font)
        plt.xticks([])
        plt.yticks([])
    plt.show()

