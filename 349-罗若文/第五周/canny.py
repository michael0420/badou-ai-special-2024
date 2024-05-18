import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import cv2


class Canny:
    pic_path = None
    img = None  # 源图片
    img_new = None  # 高斯平滑后的图片
    img_tidu = None  # 梯度后的图片
    img_yizhi = None  # 非最大值抑制后的图片
    angle = None  # 角度

    def __init__(self, img_url):
        self.pic_path = img_url

    # 初始化
    def gray(self):
        self.img = plt.imread(self.pic_path)
        if self.pic_path[-4:] == '.png':
            self.img = self.img * 255
        self.img = self.img.mean(axis=-1)

    # 高斯平滑 sigma:高斯核参数 dim:高斯核尺寸
    def gaussian_filter(self, sigma, dim):
        # 存储高斯核
        gaussian_kernel = np.zeros([dim, dim])
        # 生成序列
        tmp = [i - dim // 2 for i in range(dim)]
        # 计算高斯核
        n1 = 1 / (2 * math.pi * sigma ** 2)
        n2 = -1 / (2 * sigma ** 2)
        for i in range(dim):
            for j in range(dim):
                gaussian_kernel[i][j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        dx, dy = self.img.shape
        # 高斯平滑后的图片
        self.img_new = np.zeros(self.img.shape)
        tmp = dim // 2
        # 边缘填充
        img_pad = np.pad(self.img, ((tmp, tmp), (tmp, tmp)), 'constant')
        for i in range(dx):
            for j in range(dy):
                self.img_new[i][j] = np.sum(img_pad[i:i + dim, j:j + dim] * gaussian_kernel)

        plt.figure(1)
        plt.imshow(self.img_new.astype(np.uint8), cmap='gray')
        plt.axis('off')

    # 梯度
    def gradient(self):
        # sobel 矩阵初始化
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        # 灰度化之后的初始图片
        dx, dy = self.img.shape
        img_tidu_x = np.zeros(self.img_new.shape)  # 存储梯度图像
        img_tidu_y = np.zeros([dx, dy])
        self.img_tidu = np.zeros(self.img_new.shape)

        # 边缘填充
        img_pad = np.pad(self.img_new, ((1, 1), (1, 1)), 'constant')
        for i in range(dx):
            for j in range(dy):
                img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
                img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
                self.img_tidu[i, j] = math.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)

        img_tidu_x[img_tidu_x == 0] = 0.00000001
        self.angle = img_tidu_y / img_tidu_x
        plt.figure(2)
        plt.imshow(self.img_tidu.astype(np.uint8), cmap='gray')
        plt.axis('off')

    # 非极大值抑制
    def non_maximum_suppression(self):
        dx, dy = self.img.shape
        self.img_yizhi = np.zeros(self.img_tidu.shape)
        for i in range(1, dx - 1):
            for j in range(1, dy - 1):
                # 在8邻域内是否要抹去做个标记
                flag = True
                # 梯度幅值的8邻域矩阵
                temp = self.img_tidu[i - 1:i + 2, j - 1:j + 2]
                # 使用线性插值法判断抑制与否
                if self.angle[i, j] <= -1:
                    num_1 = (temp[0, 1] - temp[0, 0]) / self.angle[i, j] + temp[0, 1]
                    num_2 = (temp[2, 1] - temp[2, 2]) / self.angle[i, j] + temp[2, 1]
                    if not (self.img_tidu[i, j] > num_1 and self.img_tidu[i, j] > num_2):
                        flag = False
                elif self.angle[i, j] >= 1:
                    num_1 = (temp[0, 2] - temp[0, 1]) / self.angle[i, j] + temp[0, 1]
                    num_2 = (temp[2, 0] - temp[2, 1]) / self.angle[i, j] + temp[2, 1]
                    if not (self.img_tidu[i, j] > num_1 and self.img_tidu[i, j] > num_2):
                        flag = False
                elif self.angle[i, j] > 0:
                    num_1 = (temp[0, 2] - temp[1, 2]) / self.angle[i, j] + temp[1, 2]
                    num_2 = (temp[2, 0] - temp[1, 0]) / self.angle[i, j] + temp[1, 0]
                    if not (self.img_tidu[i, j] > num_1 and self.img_tidu[i, j] > num_2):
                        flag = False
                elif self.angle[i, j] < 0:
                    num_1 = (temp[1, 0] - temp[0, 0]) / self.angle[i, j] + temp[1, 0]
                    num_2 = (temp[1, 2] - temp[2, 2]) / self.angle[i, j] + temp[1, 2]
                    if not (self.img_tidu[i, j] > num_1 and self.img_tidu[i, j] > num_2):
                        flag = False
                if flag:
                    self.img_yizhi[i, j] = self.img_tidu[i, j]
        plt.figure(3)
        plt.imshow(self.img_yizhi.astype(np.uint8), cmap='gray')
        plt.axis('off')

    # 双阈值检测，遍历所有边上的点，查看8邻域是否有可能是边的点，如果是，则放入栈中
    def double_threshold(self):
        lower_boundary = self.img_tidu.mean() * 0.5
        # 设置高阈值是低阈值的3倍
        high_boundary = lower_boundary * 3
        zhan = []

        # 不考虑最外圈
        for i in range(1, self.img_yizhi.shape[0] - 1):
            for j in range(1, self.img_yizhi.shape[1] - 1):
                # 大于等于高阈值的一定是边上的点，则入栈
                if self.img_yizhi[i, j] >= high_boundary:
                    zhan.append([i, j])
                    self.img_yizhi[i, j] = 255
                elif self.img_yizhi[i, j] <= lower_boundary:  # 小于等于低阈值的一定不是边上的点，舍去
                    self.img_yizhi[i, j] = 0

        # 如果栈不为空，则取出栈顶元素，查看其8邻域内是否有可能是边的点，如果有，则入栈，否则舍去
        while len(zhan) != 0:
            temp_1, temp_2 = zhan.pop()  # 出栈
            a = self.img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]

            if high_boundary > a[0, 0] > lower_boundary:
                zhan.append([temp_1 - 1, temp_2 - 1])  # 入栈
                self.img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 将像素点标记为边缘
            if high_boundary > a[0, 1] > lower_boundary:
                zhan.append([temp_1 - 1, temp_2])
                self.img_yizhi[temp_1 - 1, temp_2] = 255
            if high_boundary > a[0, 2] > lower_boundary:
                zhan.append([temp_1 - 1, temp_2 + 1])
                self.img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            if high_boundary > a[1, 0] > lower_boundary:
                zhan.append([temp_1, temp_2 - 1])
                self.img_yizhi[temp_1, temp_2 - 1] = 255
            if high_boundary > a[1, 2] > lower_boundary:
                zhan.append([temp_1, temp_2 + 1])
                self.img_yizhi[temp_1, temp_2 + 1] = 255
            if high_boundary > a[2, 0] > lower_boundary:
                zhan.append([temp_1 + 1, temp_2 - 1])
                self.img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            if high_boundary > a[2, 1] > lower_boundary:
                zhan.append([temp_1 + 1, temp_2])
                self.img_yizhi[temp_1 + 1, temp_2] = 255
            if high_boundary > a[2, 2] > lower_boundary:
                zhan.append([temp_1 + 1, temp_2 + 1])
                self.img_yizhi[temp_1 + 1, temp_2 + 1] = 255

        for i in range(self.img_yizhi.shape[0]):
            for j in range(self.img_yizhi.shape[1]):
                if self.img_yizhi[i, j] != 0 and self.img_yizhi[i, j] != 255:
                    self.img_yizhi[i, j] = 0

        # 绘图
        plt.figure(4)
        plt.imshow(self.img_yizhi.astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    # 自定义canny
    canny = Canny('lenna.png')
    canny.gray()
    canny.gaussian_filter(0.5, 5)
    canny.gradient()
    canny.non_maximum_suppression()
    canny.double_threshold()

    # 调用canny方法
    img = cv2.imread("lenna.png", 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("canny", cv2.Canny(gray, 200, 300))
    cv2.waitKey()
    cv2.destroyAllWindows()