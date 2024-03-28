import cv2
import numpy as np
import matplotlib.pyplot as plt
# 第二周作业 1实现灰度化和二值图、2实现最临近插值
# -------------------------------------------------------------------------------------------------------------------------
# 实现灰度化函数 RGB转灰度公式：Grey = 0.299*R + 0.587*G + 0.114*B
def getGraph(path):
    img = cv2.imread(path)
    h,w = img.shape[:2]
    graph_img = np.zeros(img.shape,img.dtype)  # dtype数据类型 例如：int float
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            graph_img[i,j] = (m[0] * 0.114 + m[1] * 0.587 + m[2] * 0.299)

    return graph_img


# 实现二值化函数
def getBinary(path):
    binary_img = getGraph(path)/255
    h,w = binary_img.shape[:2]
    for i in range(h):
        for j in range(w):
            m = binary_img[i,j]
            if m > 0.5:
                binary_img[i,j] = 1
            else:
                binary_img[i,j] = 0

    return binary_img

# 实现临近插值函数
def getNewImg(path,h,w):
    img = cv2.imread(path)
    h1,w1,ch = img.shape
    newImg = np.zeros((h,w,ch),np.uint8)
    ht = h1/h
    wt = w1/w
    for i in range(h):
        for j in range(w):
            x = int(i * ht + 0.5)
            y = int(j * wt + 0.5)
            newImg[i,j] = img[x,y]

    return newImg


# -------------------------------------------------------------------------------------------------------------------------
# 调用函数
path = "lenna.png"
# 展示原图
plt.subplot(221)
img = cv2.imread(path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # opencv读取照片得到像素为BGR
plt.imshow(img)

# 展示灰度图
graph_img = getGraph(path)
plt.subplot(222)
plt.imshow(graph_img)

# 展示二值图
binary_img = getBinary(path)
plt.subplot(223)
plt.imshow(binary_img)
plt.show()

# 调用临近插值函数
new_img = getNewImg(path,200,200)
cv2.imshow('show new_img',new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
