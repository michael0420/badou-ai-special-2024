# 透视转换

import cv2
import numpy as np


class PerspectiveTransform:
    img_url = None  # 源图片路径
    src_path = []  # 源图片顶点坐标
    dst_path = [[0, 0], [0, 488], [337, 488], [337, 0]]  # 目标图片顶点坐标
    img = None

    def __init__(self, img_url):
        self.img_url = img_url

    # 获取顶点坐标
    def get_src_path(self):
        self.img = cv2.imread(self.img_url)
        # 图片灰度化
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # 高斯平滑
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 构建3*3的矩形元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # 对图像进行膨胀操作
        dilate = cv2.dilate(blurred, kernel)
        # 边缘检测
        edges = cv2.Canny(dilate, 30, 120, 3)

        # 轮廓检测
        contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = contours[0]

        docCnt = None
        if len(cnts) > 0:
            # 根据轮廓面积从大到小排序
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                # 获取轮廓周长
                peri = cv2.arcLength(c, True)
                # 获取轮廓近似
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # 如果近似的顶点个数是4个，说明找到了纸张的轮廓
                if len(approx) == 4:
                    docCnt = approx
                    break

        if docCnt is not None:
            # 获取顶点坐标
            srcArray = []
            for peak in docCnt:
                peak = peak[0]
                srcArray.append(list(peak))

            self.src_path = srcArray

    def perspective_transform(self):

        print(self.src_path)
        # 获取顶点坐标
        result3 = self.img.copy()
        # 生成透视变换矩阵
        src = np.float32(self.src_path)
        dst = np.float32(self.dst_path)
        m = cv2.getPerspectiveTransform(src, dst)
        result = cv2.warpPerspective(result3, m, (337, 488))
        cv2.imshow('src', self.img)
        cv2.imshow('result', result)
        cv2.waitKey(0)


if __name__ == '__main__':
    pt = PerspectiveTransform('photo1.jpg')
    pt.get_src_path()
    pt.perspective_transform()