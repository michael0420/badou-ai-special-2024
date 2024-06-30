import cv2

# 均值hash
def ahash(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ahash = cv2.resize(img_gray,(8,8), interpolation=cv2.INTER_AREA)

    img_sum = 0
    ahash_str = ''
    for i in range(img_ahash.shape[0]):
        for j in range(img_ahash.shape[1]):
            img_sum = img_sum + img_ahash[i,j]
    # print(img_sum)
    img_mean = img_sum/64

    for i in range(img_ahash.shape[0]):
        for j in range(img_ahash.shape[1]):
            if img_ahash[i,j] > img_mean:
                ahash_str = ahash_str +'1'
            else:
                ahash_str = ahash_str + '0'
    print(ahash_str)
    return  ahash_str

# 差值hash
def dhash(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_dhash = cv2.resize(img_gray, (9, 8), interpolation=cv2.INTER_AREA)

    dhash_str = ''

    for i in range(img_dhash.shape[0]):
        for j in range(img_dhash.shape[1]-1):
            if img_dhash[i, j] > img_dhash[i,j+1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    print(dhash_str)
    return dhash_str

# 两图hash值比较
def compare(img1,img2):
    n = 0
    if len(img1) != len(img2):
        raise ValueError
    else:
        for i in range(len(img1)):
            if img1[i] != img2[i] :
                n += 1
    return n

img1 = cv2.imread("lenna.png")
img2 = cv2.imread("lenna_GaussianNoise.png")

# 均值hash实例比对结果
img_1_ahash = ahash(img1)
img_2_ahash = ahash(img2)
n_ahash = compare(img_1_ahash, img_2_ahash)
print('均值哈希比对结果：',n_ahash)

# 差值hash实例比对结果
img_1_dhash = dhash(img1)
img_2_dhash = dhash(img2)
n_dhash = compare(img_1_dhash,img_2_dhash)
print('差值哈希比对结果：',n_dhash)
