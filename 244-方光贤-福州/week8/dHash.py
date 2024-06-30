import cv2

def dHash(img):
    #缩放为9*8 使用三次立方插值 这是为了做差值
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    #灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 设置初始哈希值
    hash_str = ''
    for i in range (8):
        for j in range (8):
            if gray[i,j] > gray[i,j+1]:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str

def comHash(hash1, hash2):
    count = 0
    #如果hash长度不同 报错 返回-1
    if len(hash1) != len(hash2):
       return -1
    else:
        for i in range(len(hash1)):
            # 不相等则计数+1，count最终为相似度
            if hash1[i] != hash2[i]:
                count = count + 1
    return count

img1 = cv2.imread('lenna.png')
img2 = cv2.imread('noisy_lenna.png') #利用之前生成的椒盐噪声lenna图

hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
count = comHash(hash1,hash2)
print('均值哈希算法相似度：', count)
