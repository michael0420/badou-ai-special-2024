import cv2
import numpy as np
import matplotlib.pylab as plt


# 归一化
def Normalization(x):
    h, w, c = x.shape
    img_Nor = np.zeros((h, w, c))
    for c1 in range(c):
        for i in range(h):
            for j in range(w):
                min_1 = x[:, :, c1].min(axis=0)[j]
                max_1 = x[:, :, c1].max(axis=0)[j]
                # print(x[i, j, c1])
                img_Nor[i, j, c1] = (float(x[i, j, c1]) - min_1) / float(max_1 - min_1)
                # print(img_Nor[i, j, c1])
            print(i)
        print(c1)
    return img_Nor

img = cv2.imread('../../imgs/lenna.png')
img_Nors = Normalization(img)
print("wancheng")
cv2.imshow("img", img_Nors)


# 标准化
def z_score(x):
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    return [(j - x_mean) / s2 for j in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10,
     10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
z = z_score(l)

cs = []
for i in l:
    c = l.count(i)
    cs.append(c)

plt.plot(l, cs)
plt.plot(z, cs)
plt.show()
cv2.waitKey(0)
