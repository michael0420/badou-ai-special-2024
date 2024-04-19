import numpy as np
import cv2

'''
实现双线性插值
'''

def bilinear(img, out_size):
    # 原图
    src_high, src_width, channels = img.shape
    # 目标图size
    destination_high, destination_width = out_size[1], out_size[0]
    # 空图
    dst_img = np.zeros((destination_high, destination_width, 3), dtype=np.uint8)
    # 缩放规模大小
    scale_x = destination_width/src_width
    scale_y = destination_high/src_high
    for i in range(3):
        for destination_y in range(destination_high):
            for destination_x in range(destination_width):
                # 计算几何中心对称值    获取原图坐标虚拟点
                src_x = (destination_x+0.5)/scale_x - 0.5
                src_y = (destination_y+0.5)/scale_y - 0.5

                # 对虚拟点进行范围规划 获取像素值
                src_x0 = int(src_x)
                src_x1 = min(src_x0+1, src_width-1)  # +1是查看改近似点邻近的坐标
                src_y0 = int(src_y)
                src_y1 = min(src_y0+1, src_high-1)

                # 计算插值像素点的值
                temp_x1 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp_x2 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]

                dst_img[destination_y, destination_x, i] = int((src_y1 - src_y) * temp_x1 + (src_y - src_y0) * temp_x2)
    return dst_img

def insert_up(img,high,wide):
    height, width, channels = img.shape
    changeImage = np.zeros((high, wide, channels), np.uint8)
    change_height = height/high    #缩放倍数
    change_width = width/wide
    for i in range(high):
        for j in range(wide):
            x = int(i*change_height + 0.5)  # 先向上加到整数，再向下取整找到最邻近的点 其实就是四舍五入
            y = int(j*change_width + 0.5)
            changeImage[i, j] = img[x, y]
    return changeImage
if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    dst_img = bilinear(img,(600,600))
    cv2.imshow("original",img)
    cv2.imshow("after bilinear",dst_img)
    img = cv2.imread("lenna.png")
    result = insert_up(img, 600, 600)
    cv2.imshow("after nearest",result)
    cv2.waitKey()
    cv2.destroyAllWindows()