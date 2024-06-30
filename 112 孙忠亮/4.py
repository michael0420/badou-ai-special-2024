import numpy as np
import cv2
import random
#高斯噪声
def gaussiannoise(src,means,sigma,percetage):
    noiseimg=src
    noisenum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(noisenum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        noiseimg[randX,randY]=noiseimg[randX,randY]+random.gauss(means,sigma)
        if noiseimg[randX,randY]<=0: noiseimg[randX,randY]=0
        elif noiseimg[randX,randY]>255: noiseimg[randX,randY]=255
        return noiseimg
img=cv2.imread('lenna.png',0)
img1=gaussiannoise(img,2,4,0.8)
cv2.imshow('lenna',img)
cv2.waitKey(0)
cv2.imshow('lenna',img1)
cv2.waitKey(0)

#椒盐噪声
def jiaoyan(src,percetage):
    noiseimg=src
    noisenum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(noisenum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.random()<=0.5: noiseimg[randX,randY]=0
        else: noiseimg[randX,randY]=255
    return noiseimg
img=cv2.imread('lenna.png',0)
img2=jiaoyan(img,0.2)
cv2.imshow('jiaoyan',img2)
cv2.waitKey(0)

#pca
def pca(X, num_components):
    #中心化
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    #计算协方差矩阵
    covariance_matrix = np.cov(X_centered.T)
    #计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    #选择主成分
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_indices = sorted_indices[:num_components]
    top_eigenvectors = eigenvectors[:, top_indices]
    #转换数据
    X_pca = np.dot(X_centered, top_eigenvectors)
    return X_pca
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
p=pca(X,3)
print(p)