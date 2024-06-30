import cv2


def avgHash(img, width=8, height=8):
    """均值hash算法"""
    # 将图像缩放为8*8
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 像素和初值设置为0,hash值初始值设置为""
    s = 0
    hash_str = ""
    # 求像素和
    for i in range(height):
        for j in range(width):
            s = s + gray[i, j]

    # 求平均灰度
    avg = s / (width * height)
    # 灰度值大于平均值为1，相反则为0，生成图片的hash值
    for i in range(height):
        for j in range(width):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def diffHash(img, width=8, height=8):
    """差值hash计算"""
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ""
    # 每行的前一个像素大于后一个像素为1，反之则为0，生成哈希序列
    for i in range(height):
        for j in range(width - 1):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def comp_hash(hash1, hash2):
    """哈希值对比"""
    n = 0
    # hash长度不同则返回-1，表示计算失败
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 如果不相等，则n+1,n表示相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return 1 - n / len(hash2)


def test_avHash(img1, img2):
    """测试均值哈希算法"""
    hash1 = avgHash(img1)
    hash2 = avgHash(img2)
    return comp_hash(hash1, hash2)


def test_diffHash(img1, img2):
    """测试差值哈希算法"""
    hash1 = diffHash(img1)
    hash2 = diffHash(img2)
    return comp_hash(hash1, hash2)
