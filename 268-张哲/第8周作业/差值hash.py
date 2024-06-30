import cv2

def Difference_hash(img):
    img = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    #转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #求hash
    hash = ''
    for i in range(8):
        for j in range(8):
            if gray[i][j]>gray[i][j+1]:
                hash += '1'
            else:
                hash += '0'
    return hash
#Hash值对比
def cmpHash(hash1,hash2):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n
img1=cv2.imread('lenna.png')
img2=cv2.imread('lenna.png')
hash1= Difference_hash(img1)
hash2= Difference_hash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print('差值哈希算法相似度：',n)
