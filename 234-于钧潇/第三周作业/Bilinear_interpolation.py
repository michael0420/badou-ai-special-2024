import cv2
import numpy as np

def bilinear_interpolatio(img, outsize):
    print("img h[{0}] w[{1}]".format(img.shape[0], img.shape[1]))
    img_h, img_w, channel = img.shape
    outimg_h, outimg_w = outsize[0], outsize[1]
    out_img = np.zeros((outimg_h, outimg_w, 3), dtype=np.uint8)
    ratio_h = float(img_h)/outimg_h
    ratio_w = float(img_w)/outimg_w
    for i in range(outimg_w):
        for j in range(outimg_h):
            for c in range(channel):
                # 虚拟点
                src_x = (i + 0.5) * ratio_w - 0.5
                src_y = (j + 0.5) * ratio_h - 0.5
                # 真实点
                src_x1 = int(src_x)
                src_x2 = min(src_x1 + 1, img_w - 1)
                src_y1 = int(src_y)
                src_y2 = min(src_y1 + 1, img_h - 1)

                # 带公式 Q11(x1,y1)  Q12(x1,y2)   Q21(x2,y1)  Q22(x2,y2)  注意x.y  取像素点时不要把参数输反了
                point_r1 = (src_x2 - src_x) * img[src_y1, src_x1, c] + (src_x - src_x1) * img[src_y1, src_x2, c]
                point_r2 = (src_x2 - src_x) * img[src_y2, src_x1, c] + (src_x - src_x1) * img[src_y2, src_x2, c]
                out_img[j, i, c] = (src_y2 - src_y) * point_r1 + (src_y - src_y1) * point_r2

    return out_img

if __name__ == '__main__':
    # 原图 393x225
    img = cv2.imread("R-C2.jpg")
    dst = bilinear_interpolatio(img, (785, 550))

    cv2.imshow('image', dst)
    cv2.waitKey()
