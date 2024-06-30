import cv2
import numpy as np
import time
import os.path as path


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
    # 缩放8*8
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
    :param hash1: 感知哈希序列1
    :param hash2: 感知哈希序列2
    :return: 返回相似度
    """
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1

    return 1 - n / len(hash2)


def pHash(img_file, width=64, high=64):
    """
    感知哈希算法
    :param img_file: 图像数据
    :param width: 图像缩放后的宽度
    :param high:图像缩放后的高度
    :return:图像感知哈希序列
    """
    # 加载并调整图片为32x32灰度图片
    img = cv2.imread(img_file, 0)
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)

    # 创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img  # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize(32, 32)

    # 把二维list变成一维list
    img_list = vis1.flatten()

    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i > avg else '1' for i in img_list]

    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 32 * 32, 4)])


def hamming_dist(s1, s2):
    return 1 - sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)]) * 1. / (32 * 32 / 4)
    


def concat_info(type_str, score, time):
    temp = '%s相似度：%.2f %% -----time=%.4f ms' % (type_str, score * 100, time)
    print(temp)
    return temp


def test_diff_hash(img1_path, img2_path, loops=1000):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    start_time = time.time()

    for _ in range(loops):
        hash1 = dHash(img1)
        hash2 = dHash(img2)
        cmp_hash(hash1, hash2)

    print(">>> 执行%s次耗费的时间为%.4f s." % (loops, time.time() - start_time))


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


def test_pHash(img1_path, img2_path):
    time1 = time.time()
    hash1 = pHash(img1_path)
    hash2 = pHash(img2_path)
    n = hamming_dist(hash1, hash2)
    return concat_info("感知哈希算法", n, time.time() - time1) + "\n"


def deal(img1_path, img2_path):
    info = ''

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 计算图像哈希相似度
    info = info + test_aHash(img1, img2)
    info = info + test_dHash(img1, img2)
    info = info + test_pHash(img1_path, img2_path)
    return info


def contact_path(file_name):
    output_path = "."
    return path.join(output_path, file_name)


def main():
    data_img_name = '0216.jpg'
    data_img_name_base = data_img_name.split(".")[0]

    base = contact_path(data_img_name)
    light = contact_path("%s_light.jpg" % data_img_name_base)
    resize = contact_path("%s_resize.jpg" % data_img_name_base)
    contrast = contact_path("%s_contrast.jpg" % data_img_name_base)
    sharp = contact_path("%s_sharp.jpg" % data_img_name_base)
    blur = contact_path("%s_blur.jpg" % data_img_name_base)
    color = contact_path("%s_color.jpg" % data_img_name_base)
    rotate = contact_path("%s_rotate.jpg" % data_img_name_base)

    # 打印文件路径以调试
    print("base:", base)
    print("light:", light)
    print("resize:", resize)
    print("contrast:", contrast)
    print("sharp:", sharp)
    print("blur:", blur)
    print("color:", color)
    print("rotate:", rotate)

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


if __name__ == '__main__':
    main()

