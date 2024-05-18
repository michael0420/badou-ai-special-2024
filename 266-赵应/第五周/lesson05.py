import cv2
import numpy as np


def get_warp_matrix(src_dem, des_dem):
    """手动实现获取变换矩阵"""
    [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = src_dem
    [(x4, y4), (x5, y5), (x6, y6), (x7, y7)] = des_dem
    matrix_a = np.array([[x0, y0, 1, 0, 0, 0, -x0 * x4, -y0 * x4],
                         [0, 0, 0, x0, y0, 1, -x0 * y4, -y0 * y4],
                         [x1, y1, 1, 0, 0, 0, -x1 * x5, -y1 * x5],
                         [0, 0, 0, x1, y1, 1, -x1 * y5, -y1 * y5],
                         [x2, y2, 1, 0, 0, 0, -x2 * x6, -y2 * x6],
                         [0, 0, 0, x2, y2, 1, -x2 * y6, -y2 * y6],
                         [x3, y3, 1, 0, 0, 0, -x3 * x7, -y3 * x7],
                         [0, 0, 0, x3, y3, 1, -x3 * y7, -y3 * y7]])
    matrix_b = np.array([[x4, y4, x5, y5, x6, y6, x7, y7]]).T
    # 计算变换系数
    matrix_transform = np.matmul(np.linalg.inv(matrix_a), matrix_b)
    # matrix_transform = np.mat(matrix_a).I * matrix_b
    # 根据变换系数生成变换矩阵
    wrap_matrix = matrix_transform.T[0]
    wrap_matrix = np.append(wrap_matrix, 1)
    wrap_matrix = wrap_matrix.reshape((3, 3))
    return wrap_matrix;


def getWarpPerspective(src_img, wrap_matrix, des_dem):
    """使用cv接口获取变换矩阵"""
    dst_img = cv2.warpPerspective(src_img, wrap_matrix, des_dem)
    cv2.imshow("warp perspective", dst_img)
    cv2.waitKey(0)


def canny(img_path):
    """使用cv接口实现canny"""
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    des_img = cv2.Canny(img_gray, 200, 300)
    cv2.imshow(" canny", des_img)
    cv2.waitKey(0)




if __name__ == '__main__':
    # src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    # src = np.array(src)
    # dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    # dst = np.array(dst)
    # des_img = get_warp_matrix(src, dst)
    # print(des_img)
    # src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    # src = np.float32([[377, 40], [742, 41], [378, 639], [742, 639]])
    # dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    # dst = np.float32([[0, 0], [365, 0], [0, 599], [365, 599]])
    # src_img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
    # wrapMatrix = cv2.getPerspectiveTransform(src, dst)
    # wrapMatrix = get_warp_matrix(src, dst)
    # getWarpPerspective(src_img, wrapMatrix, (365, 599))
    canny("photo1.jpg")
