import cv2

#均值哈希
def ahash(img):
    #缩放图片大小为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #计算图像像素均值
    avg = gray.mean()
    #遍历像素大于均值记为1，小于均值记为0
    str = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                str = str + '1'
            else:
                str = str + '0'
    return str

#差值哈希
def dhash(img):
    # 缩放图片大小为8*9
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 遍历像素，每行前一像素大于后一像素记为1，否则记为0
    str = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j] > gray[i,j+1]:
                str = str + '1'
            else:
                str = str + '0'
    return str

#哈希值比较
def cmphash(hash1, hash2):
    if len(hash1) != len(hash2):
        return -1
    else:
        n=0
        for i in range(len(hash1)):
            if hash1[i] != hash2[i]:
                n += 1
        return n

img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
ahash1 = ahash(img1)
ahash2 = ahash(img2)
print(ahash1)
print(ahash2)
print('均值哈希相似度：', cmphash(ahash1, ahash2))
dhash1 = dhash(img1)
dhash2 = dhash(img2)
print(dhash1)
print(dhash2)
print('差值哈希相似度：', cmphash(dhash1, dhash2))