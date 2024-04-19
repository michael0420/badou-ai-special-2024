import cv2
import numpy as np

# 最临近插值法（编写函数）
def func(img,dim):
    h, w, channels = img.shape
    des_h, des_w = dim[0], dim[1]
    des_img = np.zeros((des_h, des_w, channels), dtype=np.uint8)
    h_rate = des_h / h
    w_rate = des_w / w
    for i in range(des_h):
        for j in range(des_w):
            y = min(h - 1, int(i / h_rate + 0.5))
            x = min(w - 1, int(j / w_rate + 0.5))
            des_img[i, j] = img[y, x]

    return des_img

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    des_image = func(img, (700, 700))
    print(des_image.shape)
    cv2.imshow('image show des_image', des_image)
    cv2.imshow('image show lenna', img)
    cv2.waitKey(0)




# 调用函数
if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    new_width = 800
    new_height = 800
    des_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Resized Image', des_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


