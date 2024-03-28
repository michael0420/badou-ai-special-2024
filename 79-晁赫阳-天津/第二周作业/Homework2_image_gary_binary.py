import cv2
import numpy as np
import matplotlib.pyplot as plt

def greybinary(img):
    # 原图
    img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 灰度图转化
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # h,w = img.shape[0:2]
    # img_grey = np.zeros([h,w], img.dtype)
    # for i in range(h):
    #     for j in range(w):
    #         m =img[i,j]
    #         img_grey[i,j] = int(m[0]*0.114 + m[1]*0.587 +m[2]*0.299)
    # print("Image shows gray: %"%img_grey)

    # 二值图转化
    img_binary = np.where(img_grey>=128, 255, 0)
    # rows, cols = img_grey.shape
    # for i in range(rows):
    #     for j in range(cols):
    #         if (img_binary[i,j] <= 128):
    #             img_binary[i,j] = 0
    #         else:
    #             img_binary[i,j] = 255

    # 输出原图
    plt.subplot(221)
    plt.imshow(img_original)
    plt.title('Original Image')
    # 输出灰度图
    plt.subplot(222)
    plt.imshow(img_grey, cmap = 'gray')
    plt.title('Grayscale Image')
    # 输出二值图
    plt.subplot(223)
    plt.imshow(img_binary, cmap = 'gray')
    plt.title('binary image')

    plt.show()

def bilinear_interpolation(img, objective_height, objective_width):
    height, width, channels  = img.shape
    print("图片长度为：" + str(height) + "图片宽度为："+ str(width))
    EmptyImage = np.zeros((objective_height, objective_width, channels), np.uint8)
    sh = objective_height/height
    sw = objective_width/width
    for i in range(objective_height):
        for j in range(objective_width):
            x = int(i/sh+0.5)
            y = int(j/sw+0.5)
            EmptyImage[i,j] = img[x,y]
    return EmptyImage

if __name__ == '__main__':
    img = cv2.imread('请输入你需要调整的图片')
    greybinary(img)
    improved_img = bilinear_interpolation(img, 1300,1300)
    cv2.imshow("Original Image", img)
    cv2.imshow("Improved Image", improved_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()