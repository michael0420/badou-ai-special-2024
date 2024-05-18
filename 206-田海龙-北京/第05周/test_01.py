import numpy as np


def test_01():
    i = 5
    x = i / 2
    print(x)

    # 取整
    x = i // 2
    print(x)
    # 取余
    x = i % 2
    print(x)


def test_02():
    a = np.random.normal(0, 12, (3, 4))
    b = np.random.normal(0, 12, (3, 4))
    b[b == 0] = 0.000001
    print(a, "\n", b)
    c = a / b
    print(c)


# test_02()


def test_cur_path():
    import os

    print(__file__)
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)

    print(current_file_path, current_directory)

    import os

    current_working_directory = os.getcwd()

    print(current_working_directory)

    import inspect
    import os

    current_file_path = inspect.getframeinfo(inspect.currentframe()).filename
    current_directory = os.path.dirname(os.path.abspath(current_file_path))

    print(current_file_path, current_directory)


# test_cur_path()


def test_03():
    zhan = []
    zhan.append([1, 2])
    zhan.append([2, 3])

    while not len(zhan) == 0:
        temp1, temp2 = zhan.pop()

        print(temp1, temp2)


# test_03()


def bitwise_test():
    from utils import cv_imread, current_directory
    import cv2

    img_path = current_directory + "\\img\\lenna.png"

    # 读取彩色图像
    color_image = cv_imread(img_path)

    # 读取灰度图像（作为遮罩）
    gray_image = cv_imread(img_path, cv2.IMREAD_GRAYSCALE)

    # print(color_image.shape,gray_image.shape)

    # 将灰度图像转换为与彩色图像相同的大小
    # gray_image = cv2.resize(gray_image, color_image.shape[1::-1])

    # 执行位运算
    result_image = cv2.bitwise_and(color_image, color_image, mask=gray_image)

    # 显示结果
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# bitwise_test()


def img_test():
    img_path = r"D:\Desktop\maomi.jpg"
    import cv2

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()


# img_test()


def xuanzhuan_img():
    img_path = r"D:\Desktop\maomi.jpg"
    import cv2

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.imshow(img)

    src_h, src_w = img.shape[:2]

    dst_h = 500
    dst_w = int(dst_h * src_w / src_h + 0.5)

    # 由于要做选择，高宽互换
    temp = dst_h
    dst_h = dst_w
    dst_w = temp

    print(src_h, src_w)
    print(dst_h, dst_w)

    # 这里是坐标点，不是矩阵索引取值！！！
    src_points = np.float32([[0, 0], [0, 1706], [1279, 0], [1279, 1706]])
    src_points = np.float32([[0, 0], [1706, 0], [0, 1279], [1706, 1279]])

    dst_points = np.float32([[dst_h, 0], [0, 0], [dst_h, dst_w], [0, dst_w]])
    dst_points = np.float32([[0, 1706], [0, 0], [1279, 1706], [1279, 0]])

    dst_points = np.float32([[0, dst_h], [0, 0], [dst_w, dst_h], [dst_w, 0]])

    m = cv2.getPerspectiveTransform(src_points, dst_points)
    print("warpMatrix:")
    print(m)

    # result = cv2.warpPerspective(img, m, (437, 588))
    # 转换形状：宽、高
    result = cv2.warpPerspective(img, m, (dst_w, dst_h))

    # print(result[dst_w, dst_h])

    # plt.show()
    plt.figure(2)
    plt.imshow(result)
    plt.show()


xuanzhuan_img()
