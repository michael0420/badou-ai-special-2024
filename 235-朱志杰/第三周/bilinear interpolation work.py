import cv2
import numpy as np

# 读取原图
img = cv2.imread("lenna.png")

# 获取原图宽、高、通道
org_height, org_width, org_channels = img.shape

# 用户输入缩放大小
scale_height = int(input('请输入图片高度需要缩放的值：'))
scale_width = int(input('请输入图片宽度需要缩放的值：'))

# 如果缩放大小与原图一致，展示原图
if scale_width == org_width and scale_height == org_height:
    cv2.imshow('bilinear interp', img)
    cv2.waitKey()
else:
    # 创建空的图片
    change_img = np.zeros([scale_height, scale_width, org_channels], dtype=np.uint8)
    # 计算高和宽的缩放比例
    proportion_height, proportion_width = float(scale_height/org_height), float(scale_width/org_width)
    # 遍历变化后图片高、宽、通道
    for i in range(3):
        for scale_y in range(scale_height):
            for scale_x in range(scale_width):
                # 计算要插入像素点对应原图“中心对称”像素点的值
                org_x = (scale_x+0.5)/proportion_width-0.5
                org_y = (scale_y+0.5)/proportion_height-0.5
                # 计算临近四个像素点的X和Y的值
                # x1直接使用取整可以得出
                # x2使用x1+1得出，min函数使用防止超出边界
                org_x1 = int(org_x)
                org_x2 = min(org_x1 + 1, org_width-1)
                org_y1 = int(org_y)
                org_y2 = min(org_y1 + 1, org_height-1)
                # 双线插值算法
                # temp0 = (org_x2 - org_x) * img[org_y1, org_x1, i] + (org_x - org_x1) * img[org_y1, org_x2, i]
                # temp1 = (org_x2 - org_x) * img[org_y2, org_x1, i] + (org_x - org_x1) * img[org_y2, org_x2, i]
                # change_img[scale_y, scale_x, i] = int((org_y2 - org_y) * temp0 + (org_y - org_y1) * temp1)
                change_img[scale_y, scale_x, i] = int((org_y2-org_y) * ((org_x2-org_x) * img[org_y1,org_x1,i] + (org_x-org_x1) * img[org_y1,org_x2,i]) + (org_y-org_y1) * ((org_x2-org_x) * img[org_y2,org_x1,i] + (org_x-org_x1) * img[org_y2,org_x2,i]))
    cv2.imshow('bilinear interp', change_img)
    cv2.waitKey()






