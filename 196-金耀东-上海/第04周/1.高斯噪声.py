import cv2
import numpy as np
import os

# 增加高斯噪声
def add_gauss_noise(img, mean, segma):
    img = np.array(img/255, dtype=float)
    noise = np.random.normal(mean, segma, img.shape)
    out = np.clip( img + noise , 0, 1)
    return np.uint8(out * 255)

if __name__ == "__main__":
    # 获取图像文件路径并读取图像
    img_dir = "img"
    img_filename = "lenna.jpg"
    img_path = os.path.join(img_dir, img_filename)
    bgr_img = cv2.imread(img_path)

    # 添加高斯噪声
    gauss_img_1 = add_gauss_noise(bgr_img, 0, 0.05)
    gauss_img_2 = add_gauss_noise(bgr_img, 0, 0.10)

    # 输出对比结果
    imgs = np.hstack([bgr_img, gauss_img_1, gauss_img_2])
    cv2.imshow("result", imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()