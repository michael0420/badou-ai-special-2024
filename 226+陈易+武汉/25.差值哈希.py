import cv2

""""
1. 缩放：图片缩放为8*9，保留结构，除去细节
2. 灰度化：转换为灰度图
3. 比较：像素值大于后一个像素值记作1，相反记作0
	本行不与下一行对比，每行9个像素，八个差值，有8行，总共64位
4. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）
5. 对比指纹：将两幅图的指纹对比，计算汉明距离
	两个64位的hash值有多少位是不一样的，不相同位数越少，图片越相似
"""

# 差值哈希
def subHash(img):
    # 1. 缩放
    img = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    # 2.灰度化
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 3.遍历比较，初始hash_str为空，每行后一个数比前一个数大，则+1，否则+0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[1,j] < gray[i,j+1]:
                hash_str = hash_str + "1"
            else:
                hash_str = hash_str + '0'
    return hash_str

# hash值对比
def cmpHash(hash1,hash2):
    # 两个hash值的原始差异数为0
    n = 0
    if len(hash1)!=len(hash2):
        return -1
    # 遍历判断,同位数字不相等则+1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n+1
    return n

if __name__ == '__main__':
    img1 = cv2.imread('lenna.png')
    img2 = cv2.imread('noise_gs_img.png')
    hash1 = subHash(img1)
    hash2 = subHash(img2)
    print("原图hash值：",hash1)
    print("噪声hash值：",hash2)

    n = cmpHash(hash1,hash2)
    print("均值哈希算法相似度：",n)