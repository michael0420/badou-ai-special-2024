import cv2
import random
import numpy as np
import os

# 增加椒盐噪声
def add_pepper_sault_noise(img, snr):
    heigh, width = img.shape[:2]
    for i in range( int(heigh * width * snr) ):
        x = random.randint(0, heigh-1)
        y = random.randint(0, width-1)
        if random.random() < 0.5:
            img[x, y] = 0
        else:
            img[x, y] = 255
    return img

if __name__ == "__main__":
    # 获取图像文件路径并读取图像
    img_dir = "img"
    img_filename = "lenna.jpg"
    img_path = os.path.join(img_dir, img_filename)
    bgr_img = cv2.imread(img_path)

    # 信噪比
    snr = 0.1

    # 图片分解为多个通道
    (b,g,r) = cv2.split(bgr_img)
    
    # 所有通道添加椒盐噪声
    b = add_pepper_sault_noise(img=b, snr=snr)
    g = add_pepper_sault_noise(img=g, snr=snr)
    r = add_pepper_sault_noise(img=r, snr=snr)

    # 将多个通道合并成一张图像
    out_img = cv2.merge( (b, g, r) )

    # 输出对比结果
    imgs = np.hstack([bgr_img, out_img])
    cv2.imshow("result", imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()