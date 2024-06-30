import cv2
import numpy as np



#添加高斯噪声

def add_gaussian_noises(img,mean=0, sigma=2):
    noise = np.random.normal(mean, sigma, img.shape).astype('uint8')
    noisy_img = cv2.add(img, noise)
    return noisy_img

#计算哈希值
#1. cv2.resize降低计算复杂度，意味着需要处理的像素更少，计算哈希值的过程更快
#2. 提取主要特征，缩小图像尺寸可以保留图像的整体结构和主要特征，去除细节和噪声。
def aHash(img):
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    s = 0
    hash_str =''
    #遍历像素求和
    for i in range(8):
        for j in range(8):
            s = s+gray[i,j]
    #求平均灰度：
    avg = s/64
    #灰度大于均值为1相反为0生成图片的Hash值
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

def dHash(img):
    #cv2.resize的参数要求是先列后行，所以实际插值出来的数据还是8行9列
    img = cv2.resize(img,(9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j] > gray[i,j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

def cmpHash(hash1,hash2):
    if len(hash1) != len(hash2):
        return -1
    n = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n
#def compare_hash(hash1, hash2):
    #return sum(c1 != c2 for c1,c2 in zip(hash1,hash2))
def main():
    img = cv2.imread('lenna.png')
    if img is None:
        print('Error: Image')
    #添加高斯噪声
    noisy_img = add_gaussian_noises(img)
    #计算哈希值
    hash1 = aHash(img)
    hash2 = aHash(noisy_img)
    dhash1 = dHash(img)
    dhash2 = dHash(noisy_img)

    #比较哈希值
    ahash_diff = cmpHash(hash1, hash2)
    dhash_diff = cmpHash(dhash1, dhash2)
    print(f'aDifference: {ahash_diff}')
    print(f'dDifference: {dhash_diff}')

    #显示原始图像和噪声图像
    cv2.imshow('original image',img)
    cv2.imshow('noisy image', noisy_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
