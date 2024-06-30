import  cv2


def avgHash(img, width=8, high=8):
    """
    计算图像的均值哈希值
    :param img: 输入图像，numpy.ndarray类型
    :param width: 均值哈希的宽度
    :param high: 均值哈希的高度
    :return: 均值哈希值
    """
    img=cv2.resize(img,(width,high),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=""
    sum=0
    for i in range(width):
        for j in range(high):
            sum+=gray[i,j]
    avg=sum/(width*high)
    for i in range(width):
        for j in range(high):
            if gray[i,j]>avg:
                hash_str+="1"
            else:
                hash_str+="0"

    return hash_str

def dffHash(img, width=9, high=8):
    """
    计算图像的差值哈希值
    :param img: 输入图像，numpy.ndarray类型
    :param width: 差值哈希的宽度
    :param high: 差值哈希的高度
    :return: 差值哈希值
    """
    img=cv2.resize(img,(width,high),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=""
    for i in range(width-1):
        for j in range(high):
            if gray[i,j]>gray[i,j+1]:
                hash_str+="1"
            else:
                hash_str+="0"
    return hash_str

def cmpHash(hash1,hash2):
    """
    计算汉明距离
    :param hash1: 输入图像的哈希值
    :param hash2: 输入图像的哈希值
    :return: 汉明距离
    """
    n=0
    if len(hash1)!=len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            n+=1
    return n


if __name__=="__main__":
    img1=cv2.imread("../lenna.png")
    img2=cv2.imread("../lenna_noise2.jpg")
    ahash1=avgHash(img1)
    ahash2=avgHash(img2)
    print("均值哈希值：",ahash1)
    print("均值哈希值：",ahash2)
    n=cmpHash(ahash1,ahash2)
    print("均值哈希相似度：",n)


    dhash1=dffHash(img1)
    dhash2=dffHash(img2)
    print("差值哈希值：",dhash1)
    print("差值哈希值：",dhash2)
    n=cmpHash(dhash1,dhash2)
    print("差值哈希相似度：",n)
