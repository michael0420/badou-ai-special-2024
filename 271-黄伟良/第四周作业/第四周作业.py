import random as rd
import numpy as np
import cv2
###实现高斯噪声
def GaussNoise(img, means, sigma, per):
    h, w = img.shape
    noiseNumber = int(np.floor(per * h * w))

    out_img = img.copy()

    for i in range(noiseNumber):
        randomH = rd.randint(0, h - 1)
        randomW = rd.randint(0, w - 1)

        temp = out_img[randomH][randomW] + rd.gauss(means, sigma)

        if temp < 0 :
            out_img[randomH][randomW] = 0
        elif temp > 255:
            out_img[randomH][randomW] = 255
        else:
            out_img[randomH][randomW] = temp

    return out_img


###实现椒盐噪声
def PepperNosie(img, per):
    h, w = img.shape
    noiseNumber = int(np.floor(per * h * w))

    out_img = img.copy()

    for i in range(noiseNumber):
        randomH = rd.randint(0, h - 1)
        randomW = rd.randint(0, w - 1)

        pepperNoise = rd.random()

        if pepperNoise <= 0.5 :
            out_img[randomH][randomW] = 0
        if pepperNoise > 0.5 :
            out_img[randomH][randomW] = 255

    return out_img

###实现PCA
def PCA(src_data, n_components):
    ###中心化
    meanNum = []
    cent_data = []
    src_data = np.transpose(src_data)
    for data in src_data:
        meanNum.append(np.mean(data))
    for i in range(len(src_data)):
        cent_data.append(src_data[i] - meanNum[i])

    cent_data = np.transpose(cent_data)

    ###协方差矩阵
    cov = np.dot(cent_data.T,cent_data) / len(cent_data)

    ###特征值和特征向量
    eig_val, eig_vex = np.linalg.eig(cov)
    eig_vex = np.transpose(eig_vex)

    ###转换矩阵
    U = []
    sortNum = np.argsort(-eig_val)
    for i in range(len(sortNum)):
        if sortNum[i] < n_components:
            U.append(eig_vex[i])
    U = np.transpose(U)

    return np.dot(cent_data, U)



# gray_img = cv2.imread('lenna.png', 0)
# gaussImg = GaussNoise(gray_img, 2, 4, 0.8)
# pepperImg = PepperNosie(gray_img, 0.2)
# cv2.imshow('gaussImg', gaussImg)
# cv2.imshow('pepperImg', pepperImg)
# cv2.waitKey(0)
a = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
print(PCA(a,2))