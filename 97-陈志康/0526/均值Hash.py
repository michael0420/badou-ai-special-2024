# 均值哈希算法
# 步骤
# 1. 缩放：图片缩放为8*8，保留结构，除去细节。
# 2. 灰度化：转换为灰度图。
# 3. 求平均值：计算灰度图所有像素的平均值。
# 4. 比较：像素值大于平均值记作1，相反记作0，总共64位。
# 5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
# 6. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样的，不
# 相同位数越少，图片越相似
import cv2


# 均值哈希算法
def avgHash(img):
    # 1. 缩放：缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 2. 灰度化：转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 3. 求平均值：求平均灰度
    avg = s / 64
    # 4. 比较：灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            # 5. 生成hash：
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 6. 对比指纹：Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


img1 = cv2.imread('../0327/lenna.png')
img2 = cv2.imread('../0327/lenna_blur.jpg')
hash1 = avgHash(img1)
hash2 = avgHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n)
