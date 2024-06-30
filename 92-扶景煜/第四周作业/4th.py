import numpy as np
import cv2
import sys
import multiprocessing
import threading
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris
from numpy import shape
import random
flag_start = True
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def grayscale(img):  # 灰度化
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_image


def binarize(img, threshold):  # 二值化
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度图像
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def magnify(img, height, width):  # 最邻近插值法放大

    h, w, c = img.shape

    empty_image = np.zeros((height, width, c), np.uint8)
    sh = height / h
    sw = width / w
    for i in range(height):
        for j in range(width):
            x = int(i / sh + 0.5)
            y = int(j / sh + 0.5)
            empty_image[i, j] = img[x, y]
    return empty_image


def bilinear_interpolation(img, new_height, new_width):
    height, width, channels = img.shape
    new_img = np.zeros((new_height, new_width, channels), np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            x = (i + 0.0) / new_height * height
            y = (j + 0.0) / new_width * width
            x1 = int(x)
            y1 = int(y)
            x2 = min(x1 + 1, height - 1)
            y2 = min(y1 + 1, width - 1)
            u = x - x1
            v = y - y1
            for k in range(channels):
                new_img[i, j, k] = (1 - u) * (1 - v) * img[x1, y1, k] + u * (1 - v) * img[x2, y1, k] + \
                                   (1 - u) * v * img[x1, y2, k] + u * v * img[x2, y2, k]
    return new_img


def histograme_qualization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
    plt.figure()
    plt.hist(dst.ravel(), 256)
    # plt.show()
    return gray, dst


def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv2.imshow("absX", absX)
    cv2.imshow("absY", absY)
    cv2.imshow("Result", dst)
    return absX, absY, dst


def start_thread():
    count = 0
    count += 1
    if count > 1 and cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 0:
        while True:
            user_input = input("Enter a value: ")
            if user_input.upper() == "STOP":
                print('程序已终止！')
                sys.exit()
            else:
                thread = threading.Thread(target=stop, args=(user_input,))
                thread.daemon = True  # 设置为守护线程，使得主线程结束时后台线程也会结束
                thread.start()


def GaussianNoise(src, means, sigma, percentage):
    NoiseImg = src
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 高斯噪声图片边缘不处理，故-1
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        # 此处在原有像素灰度值上加上随机数
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


def fun1(src, percentage):
    NoiseImg = src
    NoiseNum = int(percentage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


def aPCA():
    x, y = load_iris(return_X_y=True)  # 加载数据，x表示数据集中的属性数据，y表示数据标签
    pca = dp.PCA(n_components=2)  # 加载pca算法，设置降维后主成分数目为2
    reduced_x = pca.fit_transform(x)  # 对原始数据进行降维，保存在reduced_x中

    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(reduced_x)):  # 按鸢尾花的类别将降维后的数据点保存在不同的表中
        if y[i] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()


class CPCA(object):
    '''用PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''

    def __init__(self, X, K):
        '''
        :param X,样本矩阵X
        :param K,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    def _centralized(self):
        '''矩阵X的中心化'''
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean  ##样本集的中心化
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def _cov(self):
        '''求样本矩阵X的协方差矩阵C'''
        # 样本集的样例总数
        ns = np.shape(self.centrX)[0]
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(
            self.C)  # 特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


while True:
    img = cv2.imread("lenna.png")
    print("0. EXIT\n1. 灰度化插值法\n2. 二值化插值法\n3. 最邻近插值法放大法\n4. 双线性插值放大法     \
        \n5. 直方图均衡法\n6. Sobel边缘检测法\n7. 高斯噪声\n8. 椒盐噪声\n9.PCA算法降维 ")

    try:
        method = int(input('请选择您的方法：'))
    except ValueError:
        print("输入错误，请输入一个整数作为方法选择")
        continue

    if method == 0:
        flag_start = False
        break
    elif method == 1:
        flag_start = False
        print('您选择了灰度化插值法，请稍等！')
        gray_result = grayscale(img)
        cv2.imshow("Gray", gray_result)
        cv2.waitKey(0)
        continue
    elif method == 2:
        flag_start = False
        print('您选择了二值化插值法，请稍等！')
        threshold_value = 127  # 设置阈值
        binary_result = binarize(img, threshold_value)
        cv2.imshow("Binary", binary_result)
        cv2.waitKey(0)
        continue
    elif method == 3:
        print('您选择了最邻近插值法放大法，请稍等！')
        magnify_result = magnify(img, 800, 800)
        cv2.imshow("Magnify", magnify_result)
        cv2.waitKey(0)
    elif method == 4:
        print('您选择了双线性插值放大法，请稍等一会！')
        dst = bilinear_interpolation(img, 700, 700)
        cv2.imshow('Bilinear Interp', dst)
        cv2.waitKey(0)
        continue
    elif method == 5:
        print("您选择了直方图均衡法，请稍等一会！")
        gray, dst = histograme_qualization(img)
        cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
        plt.show()
        cv2.waitKey(0)
        continue
    elif method == 6:
        print("您选择了Sobel边缘检测法，请稍等一会！")
        sobel(img)
        cv2.waitKey(0)
    elif method == 7:
        print("您选择了高斯噪声，请稍等一会！")
        img = cv2.imread('lenna.png', 0)
        img1 = GaussianNoise(img, 2, 4, 0.8)
        img = cv2.imread('lenna.png')
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('lenna_GaussianNoise.png',img1)
        cv2.imshow('source', img2)
        cv2.imshow('lenna_GaussianNoise', img1)
        cv2.waitKey(0)
    elif method ==8:
        print("您选择了椒盐噪声，请稍等一会！")
        img = cv2.imread('lenna.png', 0)
        img1 = fun1(img, 0.8)
        # 在文件夹中写入命名为lenna_PepperAndSalt.png的加噪后的图片
        # cv2.imwrite('lenna_PepperAndSalt.png',img1)
        img = cv2.imread('lenna.png')
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('source', img2)
        cv2.imshow('lenna_PepperAndSalt', img1)
        cv2.waitKey(0)
    elif method ==9:
        flag = True
        while flag:
            way = int(input("请选择数据集:\n1.鸢尾花\n2.样本矩阵\n3.构造矩阵\n0.EXIT"))
            if way==1:
                aPCA()
            elif way ==2:
                X = np.array(
                    [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1],
                     [3, 5, 83, 2]])  # 导入数据，维度为4
                pca = PCA(n_components=2)  # 降到2维
                pca.fit(X)  # 执行
                newX = pca.fit_transform(X)  # 降维后的数据
                print(newX)  # 输出降维后的数据
            elif way == 3:
                if __name__ == '__main__':
                    '10样本3特征的样本集, 行为样例，列为特征维度'
                    X = np.array([[10, 15, 29],
                                  [15, 46, 13],
                                  [23, 21, 30],
                                  [11, 9, 35],
                                  [42, 45, 11],
                                  [9, 48, 5],
                                  [11, 21, 14],
                                  [8, 5, 15],
                                  [11, 12, 21],
                                  [21, 20, 25]])
                    K = np.shape(X)[1] - 1
                    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
                    pca = CPCA(X, K)
            elif way==0:
                flag=False
            else:
                print("输入错误，请重新输入")
                cv2.waitKey(0)
    else:
        print("输入错误")
    cv2.waitKey(1000)

cv2.waitKey(0)
cv2.destroyAllWindows()
