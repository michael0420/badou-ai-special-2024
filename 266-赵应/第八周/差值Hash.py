import cv2


def sub_hash(img):
    img_scale = cv2.resize(img, (9, 8), interpolation=cv2.INTER_NEAREST)
    img_gray = cv2.cvtColor(img_scale, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(9-1):
            if img_gray[i][j] > img_gray[i][j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return  hash_str


def cal_hanming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Length of strings must be equal")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


if __name__ == '__main__':
    img1 = cv2.imread("./image/lenna.png")
    img2 = cv2.imread("./image/lenna.png")
    hash_str1 = sub_hash(img1)
    hash_str2 = sub_hash(img2)
    print(hash_str1)
    print(hash_str2)
    print(cal_hanming_distance(hash_str1, hash_str2))
 