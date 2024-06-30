import  cv2









# 均值哈希
def ahash(img):
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ' '
    for i in range(8):
        for j in range(8):
            s=s+gray[i,j]
    avg=s/64
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                hash_str=hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 插值hash算法：
def dHash(img):
    #缩放8*9
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    #转换灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=''
    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if   gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str


def cmphash(hash1, hash2):
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


img1 = cv2.imread('source/lenna.png')
img2 = cv2.imread('source/lenna_blur.jpg')

hash1 = ahash(img1)
hash2 = ahash(img2)
n = cmphash(hash1,hash2)
print("均值哈希算法相似度：",n)

hash1= dHash(img1)
hash2= dHash(img2)
print(hash1)
print(hash2)
n=cmphash(hash1,hash2)
print('差值哈希算法相似度：',n)