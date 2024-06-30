import cv2


def aHash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    sum = 0
    hh = ''
    for i in range(8):
        for j in range(8):
            sum += img[i][j]
    avg = sum / 64
    for i in range(8):
        for j in range(8):
            if img[i][j] >= avg:
                hh = hh + '1'
            else:
                hh = hh + '0'
    return hh


def compare_aHash():
    img = cv2.imread('lenna.png', 0)
    img_noise = cv2.imread('lenna_noise.png', 0)
    hh = aHash(img)
    hh_noise = aHash(img_noise)
    match = 0
    for i in range(len(hh)):
        if hh[i] == hh_noise[i]:
            pass
        else:
            match += 1
    print(f'aHash匹配度: {match}')

def dHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    hh = ''
    for i in range(8):
        for j in range(8):
            if img[i,j] > img[i,j+1]:
                hh += '1'
            else:
                hh += '0'
    return hh

def compare_dHash():
    img = cv2.imread('lenna.png',0)
    img_noise = cv2.imread('lenna_noise.png',0)
    hh = dHash(img)
    hh_noise = dHash(img_noise)
    match = 0
    for i in range(len(hh)):
        if hh[i] == hh_noise[i]:
            pass
        else:
            match += 1
    print(f'dHash匹配度：{match}')

if __name__ == '__main__':
    compare_aHash()
    compare_dHash()