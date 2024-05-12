import cv2
# import imutils
import numpy as np


def get_point_and_transform():
    img = cv2.imread("photo1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用高斯模糊对灰度图像进行平滑处理，这有助于减少图像噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow("blurred", blurred)

    # 对模糊后的图像进行膨胀操作，使用一个3x3的矩形结构元素。膨胀可以增强图像中的亮区域。
    rect_dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    # ellipse_dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # cv2.imshow("rect dilate", rect_dilate)
    # cv2.imshow("ellipse dilate", ellipse_dilate)

    low_threshold = 30
    high_threshold = 120
    sobel_size = 3
    # 边缘检测
    edged = cv2.Canny(rect_dilate, low_threshold, high_threshold, apertureSize=sobel_size)
    # cv2.imshow("edged image", edged)

    # 轮廓检测
    # 在OpenCV的不同版本中，findContours函数的返回值有所不同。在OpenCV 3.x中，它返回三个值：修改后的图像、轮廓和它们的层次结构。而在OpenCV 4.x中，它只返回轮廓和层次结构。
    # print(imutils.is_cv4())
    cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    docCnt = None
    if len(cnts) > 0:
        # 根据轮廓面积降序
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            # cv2.arcLength函数是OpenCV库中用于计算轮廓或曲线的周长（也称为弧长）的函数。这个函数对于几何分析和形状识别非常有用，因为不同形状的轮廓具有不同的周长。
            peri = cv2.arcLength(c, True)

            # cv2.approxPolyDP 是 OpenCV 库中的一个函数，用于对形状的轮廓进行近似，使其变成具有较少顶点的多边形。这个函数通常用于轮廓简化，它基于道格拉斯-普克算法（Douglas-Peucker algorithm）来减少轮廓点的数量，同时尽可能保持轮廓的形状。
            # epsilon: 这是一个表示近似精度的参数。它是从原始曲线到其近似曲线的最大距离。通常，它可以设置为轮廓周长的一个小百分比，例如 0.01 * cv2.arcLength(curve, True)
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break
    # print(docCnt)
    # print(docCnt.tolist())  # [[[207, 151]], [[16, 603]], [[344, 732]], [[518, 283]]]
    # 或者直接打开画图，鼠标悬停即可显示对应顶点像素位置

    for peak in docCnt:
        peak = peak[0]
        cv2.circle(img, tuple(peak), 8, (0, 0, 255))

    # cv2.imshow("img", img)

    dst_dim = [512, 512]
    src = [i[0] for i in docCnt.tolist()]
    print(src)  # [[207, 151], [16, 603], [344, 732], [518, 283]]
    src = np.float32(src)
    dst = np.float32([[0, 0], [0, 512], [512, 512], [512, 0]])  # 顶点位置要先大概确定后再映射

    # src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    # dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    print(img.shape)

    trans_matrix = cv2.getPerspectiveTransform(src, dst)
    print(trans_matrix)
    result = cv2.warpPerspective(img.copy(), trans_matrix, dst_dim)
    cv2.imshow("src", img)
    cv2.imshow("result",  result)

    cv2.waitKey()
    cv2.destroyAllWindows()


# 返回一个由指定形状和大小的结构元素组成的矩阵
def for_fun_structure_element():
    """
    cv2.MORPH_RECT：矩形结构元素
    cv2.MORPH_CROSS：十字形结构元素
    cv2.MORPH_ELLIPSE：椭圆形结构元素
    """
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 4))  # 4行3列
    print(rect, "\n")

    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    print(cross, "\n")

    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 6))
    print(ellipse, "\n")


if __name__ == '__main__':
    # for_fun_structure_element()
    get_point_and_transform()


