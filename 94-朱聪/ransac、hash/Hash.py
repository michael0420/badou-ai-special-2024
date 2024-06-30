import cv2
import numpy as np

# 均值哈希
# 读取两张图片

img2 = cv2.imread('lenna_noise.png')

def aHash(img):

    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC) # 缩放成8*8
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean_val = np.mean(gray) # 求平均值

    gray = gray.reshape(-1)
    result_array = np.where(gray > mean_val, 1, 0)
    result_matrix = result_array.reshape(8, 8)
    hash_str = ''.join(map(str, result_matrix.flatten())).replace('[', '').replace(']', '').replace(' ', '')

    return hash_str



def aHash_normal(img):
    # 缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
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

# 差值哈希
def dHash(img):
    # 缩放8*9
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算差值
    diff = gray[:, 1:] < gray[:, :-1]
    # 生成哈希
    hash_str = ''.join(str(int(x)) for x in diff.flatten())
    return hash_str


def dHash_normal(img):
    # 缩放8*9
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


img = cv2.imread('lenna.png')
print(aHash(img) == aHash_normal(img))
#
print(dHash(img) == dHash_normal(img))

# 具体汉明距离，可以直接用for循环判断每个元素是否相同，不同则汉明距离+1