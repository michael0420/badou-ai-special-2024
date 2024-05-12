import numpy as np
import math
import matplotlib.pyplot as plt
import cv2


# 第一步，高斯平滑
def step1_gaussian(img_path):
    # sigma = 0.8
    # dim = 5
    # gauss_filter = np.zeros([dim, dim])
    # tmp = [-2, -1, 0, 1, 2]
    # # tmp = [i - dim//2 for i in range(dim)]
    # n1 = 1 / (2 * math.pi * sigma ** 2)
    # n2 = -1 / (2 * sigma ** 2)
    # for i in range(dim):
    #     for j in range(dim):
    #         gauss_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    # print("gauss_filter", gauss_filter)
    # gauss_filter = gauss_filter / gauss_filter.sum()  # 归一化
    # print("gauss_filter", gauss_filter)
    # dx, dy = img.shape
    # img_new = np.zeros([dx, dy])
    # tmp = dim // 2
    # img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), "constant")
    # for i in range(dx):
    #     for j in range(dy):
    #         img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * gauss_filter)
    # plt.subplot(141)
    # plt.imshow(img_new.astype(np.uint8), cmap="gray")
    # plt.axis("off")

    # ==========================================
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 更快
    img_new = cv2.GaussianBlur(img, (5, 5), 0)
    # plt.figure(1)
    # plt.title("gaussian blur")
    # plt.imshow(img_new, cmap="gray")
    # plt.axis(False)

    return img_new


# 第二步，求梯度
def step2_sobel_grad(img_new):
    dx, dy = img_new.shape
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_grad_x = np.zeros(img_new.shape)
    img_grad_y = np.zeros(img_new.shape)
    img_grad = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), "constant")
    for i in range(dx):
        for j in range(dy):
            img_grad_x[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_x)
            img_grad_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_y)
            img_grad[i, j] = np.sqrt(img_grad_x[i, j]**2 + img_grad_y[i, j]**2)

    img_grad_x[img_grad_x == 0] = 0.00000001
    tan = img_grad_y / img_grad_x
    print("tan\n", tan)
    plt.figure(2)
    plt.title("gradient")
    plt.imshow(img_grad.astype(np.uint8), cmap="gray")
    plt.axis("off")

    return img_grad, tan

    # =====================================================
    # 如果需要更精确的梯度幅度，应该使用平方和平方根的方法。
    # grad_x = cv2.Sobel(img_new, cv2.CV_64F, 1, 0, ksize=3)
    # grad_y = cv2.Sobel(img_new, cv2.CV_64F, 0, 1, ksize=3)
    # img_grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    #
    # """
    # 这里使用了 np.arctan2 函数，它返回给定的 Y 值（gradient_y）和 X 值（gradient_x）的反正切值。
    # 与 np.arctan 不同，np.arctan2 考虑了 Y 和 X 的符号，因此它能够提供 -π 到 π（-180度到180度）范围内的正确角度。
    # 这意味着它可以区分所有四个象限的角度。
    # """
    # grad_direction = np.arctan2(grad_y, grad_x)
    #
    # plt.figure(2)
    # plt.title("gradient sqrt")
    # plt.imshow(img_grad, cmap="gray")
    # plt.axis(False)

    # 在实际应用中，如果对性能的要求高于精度，可以使用加权平均的方法。
    # abs_x = cv2.convertScaleAbs(grad_x)
    # abs_y = cv2.convertScaleAbs(grad_y)
    # img_grad2 = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    # # print(img_grad[:5, :5])
    #
    # plt.subplot(233)
    # plt.title("gradient addWeight")
    # plt.imshow(img_grad2, cmap="gray")
    # plt.axis(False)

    # return img_grad, grad_direction


# 第三步，非极大值抑制
# def step3_nms(img_grad, grad_direction):
def step3_nms(img_grad, angle):
    dx, dy = img_grad.shape

    """
    这里首先将梯度方向从弧度转换为度（因为大多数人对度的直观理解更强）。由于 np.arctan2 返回的值范围是 -π 到 π，转换为度后范围就是 -180 到 180。
    然后，代码通过检查 angle 数组中的负值，并给这些负值加上 180 度，将角度范围从 [-180, 180] 调整到 [0, 180]。
    这样做的原因是在边缘检测中，梯度方向是无方向的（即，它不区分从左到右还是从右到左的边缘），所以将所有角度都映射到 [0, 180] 范围内可以简化后续处理。
    """
    # angle = grad_direction / np.pi * 180
    # angle[angle < 0] += 180
    #
    # img_nms = np.zeros(img_grad.shape)
    #
    # q = 255
    # r = 255
    # # [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0]
    # for i in range(1, dx - 1):
    #     for j in range(1, dy - 1):
    #         if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
    #             q = img_grad[i, j + 1]
    #             r = img_grad[i, j - 1]
    #         elif 22.5 <= angle[i, j] < 67.5:
    #             q = img_grad[i + 1, j - 1]
    #             r = img_grad[i - 1, j + 1]
    #         elif 67.5 <= angle[i, j] < 112.5:
    #             q = img_grad[i + 1, j]
    #             r = img_grad[i - 1, j]
    #         elif 112.5 <= angle[i, j] < 157.5:
    #             q = img_grad[i + 1, j + 1]
    #             r = img_grad[i - 1, j - 1]
    #
    #         if img_grad[i, j] >= q and img_grad[i, j] >= r:
    #             img_nms[i, j] = img_grad[i, j]

    img_nms = np.zeros(img_grad.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_grad[i-1:i+2, j-1:j+2]  # 梯度幅值的8邻域矩阵
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                # num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_1 = (temp[0, 0] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                # num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                num_2 = (temp[2, 2] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_grad[i, j] > num_1 and img_grad[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_grad[i, j] > num_1 and img_grad[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_grad[i, j] > num_1 and img_grad[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                # num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_1 = (temp[0, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                # num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                if not (img_grad[i, j] > num_1 and img_grad[i, j] > num_2):
                    flag = False
            if flag:
                img_nms[i, j] = img_grad[i, j]


    plt.figure(3)
    plt.title("nms")
    plt.imshow(img_nms, cmap="gray")
    plt.axis(False)

    print("img_grad.mean(): ", img_grad.mean())
    low_threshold = img_grad.mean() * 0.5
    high_threshold = low_threshold * 3
    print("low_threshold: ", low_threshold)
    print("high_threshold: ", high_threshold)

    return img_nms, low_threshold, high_threshold


# 第四步、双阈值检测
# def step4_double_threshold(img_nms, low_threshold_ratio=0.05, high_threshold_ratio=0.15):
def step4_double_threshold(img_nms, low_threshold, high_threshold):
    # high_threshold = img_nms.max() * high_threshold_ratio
    # low_threshold = high_threshold * low_threshold_ratio
    # print(img_nms)
    print("img_nms min max: ", img_nms.min(), img_nms.max())

    dx, dy = img_nms.shape
    img_threshold = np.zeros((dx, dy))

    strong_i, strong_j = np.where(img_nms >= high_threshold)
    zeros_i, zeros_j = np.where(img_nms <= low_threshold)
    weak_i, weak_j = np.where((low_threshold < img_nms) & (img_nms < high_threshold))

    strong = np.int32(255)
    weak = np.int32(25)

    img_threshold[strong_i, strong_j] = strong
    img_threshold[weak_i, weak_j] = weak

    plt.figure(4)
    plt.title("double threshold")
    plt.imshow(img_threshold, cmap="gray")
    plt.axis(False)

    return img_threshold, weak, strong


# 第五步、边缘追踪
def step5_hysteresis(img_threshold, weak, strong):
    dx, dy = img_threshold.shape
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            if img_threshold[i, j] == weak:
                if (img_threshold[i - 1, j - 1] == strong or img_threshold[i, j - 1] == strong
                        or img_threshold[i + 1, j - 1] == strong or img_threshold[i - 1, j] == strong
                        or img_threshold[i + 1, j] == strong or img_threshold[i - 1, j + 1] == strong
                        or img_threshold[i, j + 1] == strong or img_threshold[i + 1, j + 1] == strong):
                    img_threshold[i, j] = strong
                else:
                    img_threshold[i, j] = 0

    plt.figure(5)
    plt.title("hysteresis")
    plt.imshow(img_threshold, cmap="gray")
    plt.axis(False)

    return img_threshold


if __name__ == '__main__':
    img_path = "./lenna.png"
    # img = plt.imread(img_path)
    # print(img)
    # if img_path.endswith(".png"):
    #     print("png type img")
    #     img *= 255
    # img = img.mean(axis=-1)  # 对RGB求均值进行灰度化

    img_new = step1_gaussian(img_path)
    img_grad, grad_direction = step2_sobel_grad(img_new)
    img_nms, low_threshold, high_threshold = step3_nms(img_grad, grad_direction)
    img_threshold, weak, strong = step4_double_threshold(img_nms, low_threshold, high_threshold)
    final_img = step5_hysteresis(img_threshold, weak, strong)

    plt.show()
