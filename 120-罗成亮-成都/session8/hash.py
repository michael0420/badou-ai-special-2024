import os.path as path
import time

import cv2


def aHash(img, width=8, high=8):
    """
    均值哈希算法
    :param img: 图像数据
    :param width: 图像缩放的宽度
    :param high: 图像缩放的高度
    :return:感知哈希序列
    """
    # 缩放为8*8
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]

    # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def dHash(img, width=9, high=8):
    """
    差值感知算法
    :param img:图像数据
    :param width:图像缩放后的宽度
    :param high: 图像缩放后的高度
    :return:感知哈希序列
    """
    # 缩放9*8
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，反之置为0，生成感知哈希序列（string）
    for i in range(high):
        for j in range(high):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def cmp_hash(hash1, hash2):
    """
    Hash值对比
    :param hash1: 哈希序列1
    :param hash2: 哈希序列2
    :return: 返回相似度
    """
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 使用异或计算不同位
    xor_result = format(int(hash1, 2) ^ int(hash2, 2), '08b')
    return 1 - xor_result.count('1') / len(hash2)


def hamming_dist(s1, s2):
    return 1 - sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)]) * 1. / (32 * 32 / 4)


def concat_info(type_str, score, time):
    temp = '%s相似度：%.2f %% -----time=%.4f ms' % (type_str, score * 100, time)
    print(temp)
    return temp


def test_diff_hash(img1_path, img2_path, loops=1000):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    total = 0
    for _ in range(loops):
        hash1 = aHash(img1)
        hash2 = aHash(img2)
        total += cmp_hash(hash1, hash2)
    print(f"{img1_path} 和 {img2_path}的平均相似度为：{total / loops}")


def test_aHash(img1, img2):
    time1 = time.time()
    hash1 = aHash(img1)
    hash2 = aHash(img2)
    n = cmp_hash(hash1, hash2)
    return concat_info("均值哈希算法", n, time.time() - time1) + "\n"


def test_dHash(img1, img2):
    time1 = time.time()
    hash1 = dHash(img1)
    hash2 = dHash(img2)
    n = cmp_hash(hash1, hash2)
    return concat_info("差值哈希算法", n, time.time() - time1) + "\n"


def deal(img1_path, img2_path):
    info = ''

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 计算图像哈希相似度
    info = info + test_aHash(img1, img2)
    info = info + test_dHash(img1, img2)
    return info


def contact_path(file_name):
    return path.join("./", file_name)


if __name__ == '__main__':
    data_img_name = "lenna.png"
    data_img_name_base = data_img_name.split(".")[0]

    base = contact_path(data_img_name)
    light = contact_path("%s_light.jpg" % data_img_name_base)
    resize = contact_path("%s_resize.jpg" % data_img_name_base)
    contrast = contact_path("%s_contrast.jpg" % data_img_name_base)
    sharp = contact_path("%s_sharp.jpg" % data_img_name_base)
    blur = contact_path("%s_blur.jpg" % data_img_name_base)
    color = contact_path("%s_color.jpg" % data_img_name_base)
    rotate = contact_path("%s_rotate.jpg" % data_img_name_base)

    # 测试算法的效率
    test_diff_hash(base, base)
    test_diff_hash(base, light)
    test_diff_hash(base, resize)
    test_diff_hash(base, contrast)
    test_diff_hash(base, sharp)
    test_diff_hash(base, blur)
    test_diff_hash(base, color)
    test_diff_hash(base, rotate)

    # 测试算法的精度(以base和light为例)
    deal(base, light)
