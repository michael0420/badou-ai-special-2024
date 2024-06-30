import random
import numpy as np
import matplotlib.pyplot as plt
import cv2


w = 100
h = 100


def random_draw(w, h, c):
    """
    生成随机像素，形成图像
    :param w: 宽
    :param h: 高
    :return:
    """

    img = np.zeros((w, h, c), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            for k in range(c):
                p = random.randint(0, 255)
                img[i][j][k] = p

    plt.imsave("./206-田海龙-北京/第04周/random_img.jpg", img)

    cv2.imshow("img", img)

    plt.imshow(img)
    plt.show()

    cv2.waitKey()


# random_draw(512,512,3)


def xx_f():
    """
    输入一个数，求几次方
    :return:
    """
    x = input("请输入一个数：")
    n = input("几次方：")
    x, n = int(x), float(n)
    res = x**n
    print(f"{x}**{n}={res}")
    return res


# xx_f()


def dot_test():
    """
    dot计算测试
    :return:
    """
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(a)

    b = a.T
    print(b)

    c = a.dot(b)
    print(c)

    # d=a*a
    # print(d)

    e = b.dot(a)
    print(e)


# t_test()


# def p_test():
#     A = np.array([[3, 2], [1, 4]])
#     λ = 1

#     # 生成 A - λI 矩阵
#     A_minus_λI = A - λ * np.eye(A.shape[0])

#     x = np.linalg.solve(A_minus_λI, np.zeros(A.shape[0]))

#     print(x)


# p_test()


def p_test_02():
    """
    求解协方差测试
    """
    import numpy as np

    # 假设 A 是一个 n x n 的矩阵，而 λ 是一个特征值
    A = np.array([[3, 2], [1, 4]])
    λ = 0  # 替换为实际的特征值

    # 计算 A - λI
    A_minus_λI = A - λ * np.eye(A.shape[0])  # 生成 A - λI 矩阵
    A_minus_λI = A

    # 求解 A - λI 的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A_minus_λI)

    # 特征值和特征向量是元组
    print("特征值:", eigenvalues)
    print("特征向量:", eigenvectors)

    a=2
    b=np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
    )

    # 验证相乘是不是=0
    a=a*b

    print(a)


p_test_02()


def dot_test():
    z=2
    b = np.array([[-0.89442719, 0.70710678], [0.4472136, 0.70710678]])

    b = np.array([-0.89442719 , 0.4472136])
    # b = np.array([2 , -1])
    # b = np.array([-2, 1])
    # b = np.array([[-2], [1]])
    A = np.array([[3, 2], [1, 4]])
    I = np.eye(2)

    # print(I)
    a = A - I * z

    res = a.dot(b)

    print(f"res：", res)
    b = np.array([0.70710678 , 0.70710678])


    z=5

    a = A - I * z

    res = a.dot(b)

    print(f"res：", res)

# dot_test()


def mean_test():
    A = np.array([[3, 2], [1, 6]])
    mean = np.mean(A)

    mean = [np.mean(a) for a in A.T]

    print(mean)


# mean_test()


def sort_test():
    tzz = np.array([1, 12, 3, 14, 5, 16, 7, 18, 29, 10])
    sort_index = np.argsort(-1 * tzz)
    print(sort_index)


# sort_test()


def index_test():
    a = np.arange(12)
    a = a.reshape(3, 4)
    print(a)

    # b=np.array([a[:,1]])
    b = [a[:, i] for i in range(a.shape[1]) if i < 3]
    # b=np.array(b)
    print(b)

    # b=b.T
    b = np.transpose(b)
    print(b)

    # b=b.transpose()
    # print(b)


# index_test()
    

def img_test():
    from utils import cv_imread
    img = cv_imread(r"D:\Desktop\maomi.jpg")
    print(img.shape)
    h,w,c=img.shape
    a=img[h-1,w-1,:]

    print(a)


# img_test()
    

def index_test_02():
    a = np.arange(12)
    a = a.reshape(3, 4)
    print(a)

    # b=np.array([a[:,1]])
    # b = [a[:, i] for i in range(a.shape[1]) if i < 3]
    # # b=np.array(b)
    # print(b)

    inxex_temp=[1,3,2,3]
    b=a[:,inxex_temp[:2]]
    print(b)

    # b=b.T

# index_test_02()
