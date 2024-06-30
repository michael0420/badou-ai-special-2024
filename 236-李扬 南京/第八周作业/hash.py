import cv2 as cv

#均值hash,通过缩小，并计算均值，大于为1，否则为0
def aHash(img):
    newImg = cv.resize(img, (8, 8), interpolation=cv.INTER_CUBIC)
    grayImg = cv.cvtColor(newImg, cv.COLOR_BGR2GRAY)
    sum = 0

    for i in range(8):
        for j in range(8):
            sum += grayImg[i, j]

    average = sum/64

    hashStr=''
    for i in range(8):
        for j in range(8):
            if(grayImg[i, j] > average):
                hashStr += '1'
            else:
                hashStr += '0'

    return hashStr

#差值算法，缩小成8*9,计算前后两者的插值
def dHash(img):
    newImg = cv.resize(img, (9, 8), interpolation=cv.INTER_CUBIC)
    grayImg = cv.cvtColor(newImg, cv.COLOR_BGR2GRAY)

    hashStr = ''
    for i in range(8):
        for j in range(8):
            if grayImg[i, j + 1] < grayImg[i, j]:
                hashStr += '1'
            else:
                hashStr += '0'

    return hashStr

#对比hash字符串
def cmpHash(hash1, hash2):
    n = 0
    if(len(hash1) != len(hash2)):
        return -1

    for i in range(len(hash1)):
        if(hash1[i] != hash2[i]):
            n += 1

    return n

img1 = cv.imread('lenna.png')
img2 = cv.imread('lenaNoise.png')
hash1 = aHash(img1)
print(hash1)
hash2 = aHash(img2)
print(hash2)
n = cmpHash(hash1, hash2)
print('均值算法：',n)

hash11 = dHash(img1)
print(hash11)
hash22 = dHash(img2)
print(hash22)
n = cmpHash(hash11, hash22)
print('差值算法：',n)

