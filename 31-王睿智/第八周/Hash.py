import cv2

def aHash(img):
    """
    均值哈希
    :param img:
    :return:
    """
    """
    src: 输入图像。
    dsize: 输出图像的尺寸，可以是一个单元素的元组（仅指定宽度），或者两个元素的元组（宽度和高度）。
    fx 和 fy: 缩放因子，分别表示宽度和高度的缩放比例。如果未指定，则使用dsize参数。
    interpolation: 插值方法，用于确定像素值,参数如下
    INTER_NEAREST:最邻近插值
    INTER_LINEAR:双线性插值（默认）
    INTER_CUBIC:4x4像素邻域内的双立方插值
    INTER_AREA:使用像素区域关系进行重采样
    INTER_LANCZOS4:8x8像素邻域内的Lanczos插值
    """
    cv2.resize(img, (8,8), interpolation=cv2.INTER_CUBIC)
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s 为像素和初始值0，hash_str为hash初始值''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]

    # 求平均值
    avg = s/64
    # 灰度大于平均值为1，反之为0,生成图片hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'

    return hash_str


def dHash(img):
    """
    差值算法
    :param img:
    :return:
    """
    # 缩放8*9
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，反之为0，生成图片哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

def cmpHash(hash1, hash2):
    """
    哈希值比对
    :param hash1:
    :param hash2:
    :return:
    """
    # 相似度值
    n = 0
    # hash长度不同则返回-1，表示参数错误
    if len(hash1) != len(hash2):
        return -1

    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n 计数+1
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

if __name__ == '__main__':
    img1 = cv2.imread('lenna.png')
    img2 = cv2.imread('lenna_color.jpg')
    hash1 = aHash(img1)
    hash2 = aHash(img2)
    print(hash1)
    print(hash2)
    n = cmpHash(hash1, hash2)
    print('均值哈希算法相似度：', n)

    hash1 = dHash(img1)
    hash2 = dHash(img2)
    print(hash1)
    print(hash2)
    n = cmpHash(hash1, hash2)
    print('差值哈希算法相似度：', n)
