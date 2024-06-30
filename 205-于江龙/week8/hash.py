import cv2

def aHash(img):
    # size the image to 8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # convert to gray image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hash_str = ''

    # calculate the average value of the image
    avg = sum([sum(img_gray_row) for img_gray_row in img_gray]) / 64

    # calculate the hash value
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

def dHash(img):
    # size the image to 9*8
    img = cv2.resize(img, (9,8), interpolation=cv2.INTER_CUBIC)
    # convert to gray image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hash_str = ''

    # calculate the hash value
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > img_gray[i, j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    
    return hash_str

def cmpHash(hash1, hash2):
    n=0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n

if __name__ == '__main__':
    img1 = cv2.imread('../lenna.png')
    img2 = cv2.GaussianBlur(img1, (5, 5), 0)

    hash1 = aHash(img1)
    hash2 = aHash(img2)
    n = cmpHash(hash1, hash2)
    print('aHash1:', hash1)
    print('aHash2:', hash2)
    print('均值哈希算法相似度: ', n)

    hash1 = dHash(img1)
    hash2 = dHash(img2)
    n = cmpHash(hash1, hash2)
    print('dHash1:', hash1)
    print('dHash2:', hash2)
    print('差值哈希算法相似度: ', n)