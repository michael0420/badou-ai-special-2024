import cv2
import numpy as np

def Bilinear_interpolation(image, new_size):
    # 获取图像尺寸和通道数
    h, w, c = image.shape

    # 将新图像的高宽命名为new_h,new_w
    new_h, new_w = new_size[0], new_size[1]

    # 若原图与新图尺寸相同，则直接返回原图
    if new_h == h and new_w == w:
        return image.copy()

    # 创建一个新图像
    target_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    # 计算新图与原图的缩放比例
    scale_x = float(w / new_w)
    scale_y = float(h / new_h)

    for i in range(c):
        for x in range(new_w):
            for y in range(new_h):
                # 计算新图坐标在原始图像上的坐标位置，+0.5使原图与新图中心相同
                origin_x = (x + 0.5) * scale_x - 0.5
                origin_y = (y + 0.5) * scale_y - 0.5

                # 计算离新图坐标最邻近的四个像素的坐标
                x1 = int(np.floor(origin_x))
                x2 = min(x1+1, w-1)
                y1 = int(np.floor(origin_y))
                y2 = min(y1+1, h-1)

                # 计算双线性插值
                # 计算R1和R2的像素值
                # image[h,w,c]
                R1 = (x2 - origin_x) * image[y1, x1, i] + (origin_x - x1) * image[y1, x2, i]
                R2 = (x2 - origin_x) * image[y2, x1, i] + (origin_x - x1) * image[y2, x2, i]

                # 计算新图上对应点的像素值
                target_img[y, x, i] = int((y2 - origin_y) * R1 + (origin_y - y1) * R2)

    return target_img

# 加载图像
img = cv2.imread("lenna.png")

# 执行双线性插值
interpolated_img = Bilinear_interpolation(img, (700, 800))

# 显示原图和新图
cv2.imshow('original_img', img)
cv2.imshow('interpolated_img', interpolated_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


