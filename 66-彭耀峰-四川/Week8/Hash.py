import cv2
'''
实现均值hash和差值hash
'''

#均值hash
def avgHash(img):
    #缩放为8*8
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    s=0
    for i in range(8):
        for j in range(8):
            s = s + gray[i,j]
    avg = s/64
    hash_str = ''
    #生成hash值，灰度大于平均值为1 否则为0
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def diffHash(img):
    #缩放到8*9
    img = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    print(img.shape)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j] > gray[i,j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str



def compareHash(hash1,hash2):
    #hash长度不一致直接返回
    if len(hash1)!=len(hash2):
        return -1
    n = 0   #最终相似度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n=n+1
    return n


img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
hash1 = avgHash(img1)
hash2 = avgHash(img2)
print(hash1)
print(hash2)
n = compareHash(hash1,hash2)
print('均值hash相似度：',n)

hash1 = diffHash(img1)
hash2 = diffHash(img2)
print(hash1)
print(hash2)
n = compareHash(hash1,hash2)
print('差值hash相似度：',n)
