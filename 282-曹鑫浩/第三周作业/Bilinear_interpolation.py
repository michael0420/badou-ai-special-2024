import cv2
import numpy as np
import matplotlib.pyplot as plt

def bilinear_interpolation(Original_img, Dst_img_shape):
    img_original = cv2.imread(Original_img, 0)
    img_dst = np.zeros(Dst_img_shape, img_original.dtype)
    print(img_dst)
    h, w = img_original.shape[:2]
    sh = h / Dst_img_shape[0]
    sw = w / Dst_img_shape[1]
    for i in range(Dst_img_shape[0]):
        for j in range(Dst_img_shape[1]):
            P_x = i * sh + (1- sh)/sh
            P_y = j * sw + (1- sw)/sw
            if 0 <= P_x < img_original.shape[1] - 1 and 0 <= P_y < img_original.shape[0] - 1:
                R1 = (int(P_x + 1) - P_x) * img_original[int(P_x), int(P_y + 1)] + \
                     (P_x - int(P_x)) * img_original[int(P_x + 1), int(P_y + 1)]
                R2 = (int(P_x + 1) - P_x) * img_original[int(P_x), int(P_y)] + \
                     (P_x - int(P_x)) * img_original[int(P_x + 1), int(P_y)]
                img_dst[i, j] = (int(P_y + 1) - P_y) * R2 + (P_y - int(P_y)) * R1
            else:
                img_dst[i, j] = img_original[int(P_x + 0.5), int(P_y + 0.5)]
    #         R1 = (min(int(i*sw+1),w-1) - i*sw)*img_original[int(i*sw), min(int(j*sw+1),w-1)] + (i*sw - int(i*sw))*img_original\
    #             [min(int(i*sw+1),w-1), min(int(j*sw+1),h-1)]
    #         R2 = (min(int(i*sw+1),w-1) - i*sw)*img_original[int(i*sw), int(j*sw)] + (i*sw - int(i*sw))*img_original\
    #             [min(int(i*sw+1),w-1), int(j*sw)]
    #         img_dst[i, j] = (j*sh - int(j*sh))*R1 + (min(int(j*sh+1),h-1) - j*sh)*R2
    return img_dst

img = bilinear_interpolation('lenna.png', [400,400])
img_input = cv2.imread('lenna.png',0)
plt.subplot(211)
plt.imshow(img_input, cmap=plt.cm.gray)
plt.subplot(212)
plt.imshow(img, cmap=plt.cm.gray)
plt.show()
# cv2.imshow('img_input', img_input)
# cv2.imshow('img', img)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()


