import os
import cv2
import math
import numpy as np

def display_img(winname, img):
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Canny:
    def __init__(self):
        # 原始图像
        self.img = None
        # 高斯模糊后的图像
        self.guass_blur_img = None
        # 图像梯度
        self.gradients = None
        # 梯度的方向（-pi/2 , pi/2)
        self.angles = None
        # 检测出的图像边缘
        self.edges = None

    # 使用高斯平滑消除高频噪声
    def private_guassian_blur(self):
        # 初始化高斯核参数
        sigma = 0.75
        ksize = 5
        mid = ksize//2

        # 计算高斯核
        guass_kernel = np.zeros(shape=(ksize,ksize))
        n1 = 1 / (2 * math.pi * sigma ** 2)
        n2 = -1 / (2 * sigma ** 2)
        for i in range(ksize):
            x = i - mid
            for j in range(ksize):
                y = j - mid
                guass_kernel[i, j] = n1 * math.exp(n2 * (x ** 2 + y ** 2))
        guass_kernel = guass_kernel / np.sum(guass_kernel)

        # 进行高斯滤波
        self.guass_blur_img = np.zeros(shape=self.img.shape)
        img_pad = np.pad(self.img, (mid, mid), mode="constant")
        height , width = self.img.shape
        for h in range(height):
            for w in range(width):
                self.guass_blur_img[h, w] = np.sum( img_pad[h:h+ksize, w:w+ksize] * guass_kernel )
        self.guass_blur_img = np.uint8( self.guass_blur_img )

    # 使用sobel核计算图像亮度梯度
    def private_grad(self):
        kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        kernel_y = np.array([[1 ,2, 1],[0, 0, 0],[-1, -2, -1]])
        padding_img = np.pad(self.guass_blur_img, pad_width=(1,1), mode='edge' )
        height, width = self.img.shape
        self.gradients = np.zeros(shape=[height, width])
        self.angles = np.zeros(shape=[height, width])
        for i in range(height):
            for j in range(width):
                dx = np.sum(padding_img[i:i+3, j:j+3] * kernel_x) + 0.00000001
                dy = np.sum(padding_img[i:i+3, j:j+3] * kernel_y) + 0.00000001
                self.gradients[i, j] = np.sqrt(dx ** 2 + dy ** 2)
                self.angles[i, j] = np.arctan(dy / dx)
        self.gradients = np.uint8(self.gradients)

    # 非极大值抑制，剪除虚假边缘
    def private_non_max_suppression(self):
        height, width = self.gradients.shape
        self.edges = np.zeros(shape=[height, width])
        tmp_g1 , tmp_g2 , w = 0.0 , 0.0, 0.0
        for i in range(1,height-1):
            for j in range(1,width-1):
                if self.angles[i,j] >= -np.pi/4 and self.angles[i,j] <= 0:
                    w =  -np.tan(self.angles[i,j])
                    tmp_g1 = w * self.gradients[i+1, j+1] + (1-w) * self.gradients[i, j+1]
                    tmp_g2 = w * self.gradients[i-1, j-1] + (1-w) * self.gradients[i, j-1]
                elif self.angles[i,j] >= 0 and self.angles[i,j] <= np.pi/4:
                    w = np.tan(self.angles[i,j])
                    tmp_g1 = w * self.gradients[i-1, j+1] + (1-w) * self.gradients[i, j+1]
                    tmp_g2 = w * self.gradients[i+1, j-1] + (1-w) * self.gradients[i, j-1]
                elif self.angles[i,j] >= -np.pi/2 and self.angles[i,j] < -np.pi/4:
                    w = -1/np.tan(self.angles[i,j])
                    tmp_g1 = w * self.gradients[i+1, j+1] + (1-w) * self.gradients[i+1, j]
                    tmp_g2 = w * self.gradients[i-1, j-1] + (1-w) * self.gradients[i-1, j]
                elif self.angles[i,j] > np.pi/4 and self.angles[i,j] <= np.pi/2:
                    w = 1/np.tan(self.angles[i,j])
                    tmp_g1 = w * self.gradients[i+1, j-1] + (1-w) * self.gradients[i+1, j]
                    tmp_g2 = w * self.gradients[i-1, j+1] + (1-w) * self.gradients[i-1, j]

                if self.gradients[i,j] > tmp_g1 and self.gradients[i,j] > tmp_g2:
                    self.edges[i,j] = self.gradients[i,j]

    # 双阈值检测，筛选并连接边缘
    def private_double_threshold(self):
        min_val = self.gradients.mean()
        max_val = min_val * 2
        edges_stack = []
        height , width = self.edges.shape
        for i in range(1, height-1):
            for j in range(1, width-1):
                if self.edges[i, j] > max_val:
                    self.edges[i, j] = 255
                    edges_stack.append([i, j])
                elif self.edges[i ,j] < min_val:
                    self.edges[i, j] = 0

        while len(edges_stack) > 0:
            base_h, base_w = edges_stack.pop()
            for i in range(-1,2):
                for j in range(-1,2):
                    h , w = base_h + i , base_w + j
                    if h >= height or w >= width or h < 0 or w < 0:
                        continue
                    elif self.edges[h, w] > min_val and self.edges[h, w] < max_val:
                        self.edges[h, w] = 255
                        edges_stack.append([h, w])

        for i in range(height):
            for j in range(width):
                if self.edges[i, j] != 0 and self.edges[i, j] != 255:
                    self.edges[i, j] = 0

    # 执行canny算法
    def fit(self, img):
        self.img = img
        self.private_guassian_blur()
        self.private_grad()
        self.private_non_max_suppression()
        self.private_double_threshold()
        return self.edges

if __name__ == "__main__":
    # 获取图像文件路径
    img_dir = "img"
    img_filename = "lenna.jpg"
    img_path = os.path.join(img_dir, img_filename)
    # 读取灰度图
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 加载canny算法
    my_canny = Canny()
    # 进行边缘检测
    img_edges = my_canny.fit(img_gray)

    # 展示图像
    display_img("gray", img_gray)
    display_img("edges", img_edges)
