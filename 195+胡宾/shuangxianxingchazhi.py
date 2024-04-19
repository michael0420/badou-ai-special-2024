import numpy as np
import cv2
def shuangxianxing_chazhi(picture, chang_kuan):
    org_hi, org_whith, channe = picture.shape
    target_h, target_w = chang_kuan[1], chang_kuan[0]
    if org_hi == target_h and org_whith == target_w:
        return picture.copy()

    target_img = np.zeros((target_h, target_w, channe), dtype=np.uint8)

    proportion_x = float(org_whith) / target_w
    proportion_y = float(org_hi) / target_h
    for i in range(channe):
        for y in range(target_h):
            for x in range(target_w):
                # 要使图像对称需将原图与目标图平移
                src_x = (x + 0.5) * proportion_x - 0.5
                src_y = (y + 0.5) * proportion_y - 0.5
                # 防止索引越界,floor向下取整
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, org_whith - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, org_hi - 1)
                # 带入公式进行计算
                temp_x = (src_x1-src_x) * picture[src_y0, src_x0, i] + (src_x - src_x0) * picture[src_y0, src_x1, i]
                tempx_y = (src_x1 - src_x) * picture[src_y1, src_x0, i] + (src_x - src_x0) * picture[src_y1, src_x1, i]
                target_img[y , x, i] = int((src_y1 - src_y) * temp_x + (src_y - src_y0) * tempx_y)
    return target_img


if __name__ == "__main__":
    img = cv2.imread('lenna.png')
    dst = shuangxianxing_chazhi(img, (300, 300))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()

