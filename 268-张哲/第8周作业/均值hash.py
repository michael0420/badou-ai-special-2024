import cv2

def meanhash(img):
    '''
    interpolation:
    INTER_NEAREST	最邻近插值	0
    INTER_LINEAR	双线性插值 （默认）	1
    INTER_CUBIC	4x4像素邻域内的双立方插值	2
    INTER_AREA	使用像素区域关系进行重采样	3
    INTER_LANCZOS4	8x8像素邻域内的Lanczos插值	4

    '''
    img = cv2.resize(img,(8,8),interpolation = cv2.INTER_CUBIC)
    #转为灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #求图像均值
    mean = 0
    for i in range(8):
        for j in range(8):
            mean += gray[i][j]
    mean = mean /64
    #求哈希值
    hash_mean = ''
    for i in range(8):
        for j in range(8):
            if gray[i][j]>mean:
                hash_mean += '1'
            else:
                hash_mean += '0'
    return hash_mean
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
img2=cv2.imread('photo1.jpg')
hash1= meanhash(img1)
hash2= meanhash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print('均值哈希算法相似度：',n)