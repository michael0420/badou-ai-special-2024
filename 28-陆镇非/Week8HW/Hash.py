from Utils import ImageUtils
import numpy as np
import cv2

class Hash(object):
    @staticmethod
    def getAverageHash(img, targetSize:tuple=(8,8)):
        greyImg = ImageUtils.BGRImage2GreyImage(img)
        greyImg = ImageUtils.biLinearInterpolation(greyImg, targetSize)
        avg = np.sum(greyImg)/(targetSize[0]*targetSize[1])
        hashStr = ''
        for i in range(targetSize[0]):
            for j in range(targetSize[1]):
                if(greyImg[i, j] > avg):
                    hashStr = hashStr + "1"
                else:
                    hashStr = hashStr + "0"
        return hashStr

    @staticmethod
    def getDifferenceHash(img, targetSize:tuple=(8,9)):
        greyImg = ImageUtils.BGRImage2GreyImage(img)
        greyImg = ImageUtils.biLinearInterpolation(greyImg, targetSize)
        hashStr = ""
        for i in range(targetSize[0]):
            for j in range(targetSize[1]-1):
                if(greyImg[i, j] > greyImg[i, j+1]):
                    hashStr = hashStr + "1"
                else:
                    hashStr = hashStr + "0"
        return hashStr

    @staticmethod
    def compareTwoHashStr(hash1, hash2):
        if(len(hash1) != len(hash2)):
            return -1
        n = 0
        N = len(hash1)
        for i in range(0, N):
            if(hash1[i] == hash2[i]):
                n = n + 1
        return n/N

    @staticmethod
    def getPHash(img, targetSize:tuple=(64,64)):
        img = ImageUtils.BGRImage2GreyImage(img)
        img = ImageUtils.biLinearInterpolation(img, targetSize)

        # Dct 2d
        # vis1 = cv2.dct(vis0)
        vis1 = Hash.dct_2d_optimized(img)
        vis1_Flatten = vis1.reshape(-1)
        # print(vis1_Flatten)

        # calculate average
        avg = sum(vis1_Flatten) * 1. / len(vis1_Flatten)
        avg_list = ['0' if i > avg else '1' for i in vis1_Flatten]
        # print(len(avg_list))
        # get hash str in 16th , 4 bit binary to 10th(int), then %x to 16th  :  0b1111 -> 0xf, 0b00001111 -> 0x0f
        return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 32 * 32, 4)])

    @staticmethod
    def alpha(u, N):
        return np.sqrt(1.0 / N) if u == 0 else np.sqrt(2.0 / N)

    @staticmethod
    def dct_2d(matrix):
        N = matrix.shape[0]
        dct_matrix = np.zeros((N, N), dtype=np.float32)
        for u in range(N):
            for v in range(N):
                sum = 0.0
                for x in range(N):
                    for y in range(N):
                        sum += matrix[x, y] * np.cos((2 * x + 1) * u * np.pi / (2 * N)) * np.cos(
                            (2 * y + 1) * v * np.pi / (2 * N))
                dct_matrix[u, v] = Hash.alpha(u, N) * Hash.alpha(v, N) * sum
        return dct_matrix

    @staticmethod
    def idct_2d(dct_matrix):
        N = dct_matrix.shape[0]
        idct_matrix = np.zeros((N, N), dtype=np.float32)
        for x in range(N):
            for y in range(N):
                sum = 0.0
                for u in range(N):
                    for v in range(N):
                        sum += Hash.alpha(u, N) * Hash.alpha(v, N) * dct_matrix[u, v] * np.cos(
                            (2 * x + 1) * u * np.pi / (2 * N)) * np.cos((2 * y + 1) * v * np.pi / (2 * N))
                idct_matrix[x, y] = sum
        return idct_matrix

    @staticmethod
    def dct_matrix(N):
        C = np.zeros((N, N), dtype=np.float32)
        for u in range(N):
            for x in range(N):
                C[u, x] = Hash.alpha(u, N) * np.cos((2 * x + 1) * u * np.pi / (2 * N))
        return C

    @staticmethod
    def dct_2d_optimized(matrix):
        N = matrix.shape[0]
        C = Hash.dct_matrix(N)
        return np.dot(C, np.dot(matrix, C.T))

    @staticmethod
    def idct_2d_optimized(dct_matrix):
        N = dct_matrix.shape[0]
        C = dct_matrix(N)
        return np.dot(C.T, np.dot(dct_matrix, C))