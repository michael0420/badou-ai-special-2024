
import numpy as np
import cv2

'''
Implement bilinear interpolation

'''
def bilinear_interpolation(img,dim):
    ori_h, ori_w ,channel = img.shape[:3]
    des_h, des_w = dim[0], dim[1]
    print('ori_h,ori_w', ori_h, ori_w)
    print('des_h,des_w', des_h, des_w)

    if ori_h==des_h and ori_w==des_w :
        return img.copy()
    des_img = np.zeros((des_h, des_w, channel), dtype=np.uint8)
    h_rate, w_rate =float(ori_h )/ des_h, float(ori_w) / des_w
    for i in range(channel):
        for des_y in range(des_h):
            for des_x in range(des_w):
                # 利用目标图像坐标找到原图像坐标
                # 以几何中心点为图像重合点
                ori_x = (des_x + 0.5) * w_rate - 0.5
                ori_y = (des_y + 0.5) * h_rate - 0.5

                # 周围4个坐标点取值
                ori_x0 = int(np.floor(ori_x))
                ori_x1 = min(int(ori_x + 1), ori_w - 1)
                ori_y0 = int(np.floor(ori_y))
                ori_y1 = min(int(ori_y + 1), ori_h - 1)

                # 四个点带入公式计算
                dx = ori_x - ori_x0
                dy = ori_y - ori_y0
                # 在x轴方向插值
                tx0 = (1 - dx) * img[ori_y0, ori_x0, i] + dx * img[ori_y0, ori_x1, i]
                tx1 = (1 - dx) * img[ori_y1, ori_x0, i] + dx * img[ori_y1, ori_x1, i]
                # 在x轴方向插值
                des_img[des_y, des_x, i] = int((1 - dy) * tx0 + dy * tx1)
    return des_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    des = bilinear_interpolation(img, (800, 800))
    cv2.imshow('des_img', des)
    cv2.waitKey(0)
    cv2.destroyAllWindows()