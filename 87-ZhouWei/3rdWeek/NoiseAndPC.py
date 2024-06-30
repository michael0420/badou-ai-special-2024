#高斯噪声， 椒盐噪声， pca, 证明中心化协方差矩阵公式
import random
import matplotlib.pyplot as plt
import sklearn.decomposition as sd
import numpy as np
#from sklearn.datasets.base import load_iris
from sklearn.datasets import load_iris
import cv2

def gaussNoise(img, mean, sigma, percentage):
    Noiseimg = img
    Noisenum = int(percentage * img.shape[0]*img.shape[1])
    for i in range(Noisenum):
        randomX =  random.randint(0, img.shape[0]-1)
        randomY = random.randint(0, img.shape[1]-1)
        Noiseimg[randomX, randomY] += random.gauss(mean, sigma)
        if Noiseimg[randomX, randomY] < 0:
            Noiseimg[randomX, randomY] = 0
        if Noiseimg[randomX, randomY] > 255:
            Noiseimg[randomX, randomY] = 255
    return Noiseimg

def pepersaltNoise(img, percentage):
    noiseImg = img
    noiseNum = int(percentage * img.shape[0]*img.shape[1])
    for i in range(noiseNum):
        randomX = random.randint(0, img.shape[0] - 1)
        randomY = random.randint(0, img.shape[1] - 1)
        if random.random() <= 0.5:
            noiseImg[randomX, randomY] = 0
        else:
            noiseImg[randomX, randomY] = 255
    return noiseImg

# PCA
def PCALIB(x):
    pca = sd.PCA(n_components=2)
    pca.fit(x)
    return pca.fit_transform(x)

def PCASelfDef(data, components):
    #features = data.shape[1]
    data = data - data.mean(axis=0)
    print(data)

    covariance = np.dot(data.T, data)/data.shape[0]
    eig_vals,eig_vectors = np.linalg.eig(covariance)
    idx = np.argsort(-eig_vals)
    components = eig_vectors[:, idx[:components]]
    return np.dot(data, components)

img = cv2.imread('../lenna.png', 0)

fig = plt.figure(figsize=(8, 7))
fig.add_subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")

gaussimg = gaussNoise(img, 5, 14, 1)
fig.add_subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(gaussimg, cv2.COLOR_BGR2RGB))
plt.title("gaussNoise")

psimg = pepersaltNoise(img, 1)
fig.add_subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(psimg, cv2.COLOR_BGR2RGB))
plt.title("pepersaltNoise")

iris_data = load_iris()
x = iris_data.data
y = iris_data.target
reduced_x = PCALIB(x)

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_x)):
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    elif y[i] == 2:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

pca1 = fig.add_subplot(2, 3, 4)
pca1.scatter(red_x, red_y, c='r', marker='x')
pca1.scatter(green_x, green_y, c='g', marker='.')
pca1.scatter(blue_x, blue_y, c='b', marker='*')
pca1.set_title("PCV")

reduced_x1 = PCASelfDef(iris_data.data, 2)

red_x1, red_y1 = [], []
blue_x1, blue_y1 = [], []
green_x1, green_y1 = [], []
for i in range(len(reduced_x1)):
    if y[i] == 0:
        red_x1.append(reduced_x[i][0])
        red_y1.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x1.append(reduced_x[i][0])
        blue_y1.append(reduced_x[i][1])
    elif y[i] == 2:
        green_x1.append(reduced_x[i][0])
        green_y1.append(reduced_x[i][1])

pca2 = fig.add_subplot(2, 3, 5)
pca2.scatter(red_x1, red_y1, c='r', marker='x')
pca2.scatter(green_x1, green_y1, c='g', marker='.')
pca2.scatter(blue_x1, blue_y1, c='b', marker='*')
pca2.set_title("PCV_SELF_DEF")

plt.show()
