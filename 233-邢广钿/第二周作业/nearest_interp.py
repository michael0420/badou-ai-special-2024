import cv2
import numpy as np


# ===图片缩放===img：原图 high：缩放后的高 wide：缩放后的宽
def img_scaling(img, new_high, new_wide):
    # 获取原图的高、宽、通道值
    source_high, source_wide, source_channels = img.shape
    # 按新图的长宽创建新空白图
    new_img = np.zeros((new_high, new_wide, source_channels), np.uint8)
    print("新高:%s,新宽:%s", new_high, new_wide)
    # 计算原图跟旧图的宽高比
    sh = new_high / source_high
    sw = new_wide / source_wide
    for i in range(new_high):
        for j in range(new_wide):
            # 根据宽高比 计算当前新图像素点取原图像素点的位置 int:向下取整  +0.5等同于四舍五入
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            new_img[i, j] = img[x, y]
    return new_img


img = cv2.imread("lenna.png")
new_img1 = img_scaling(img, 800, 800)
new_img2 = img_scaling(img, 300, 300)
cv2.imshow("new_img1", new_img1)
cv2.imshow("new_img2", new_img2)
cv2.imshow("source", img)
cv2.waitKey(0)
