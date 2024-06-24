"""
Hash algorithm for image similarity comparison
1 - Average Hash (aHash)
2 - Difference Hash (dHash) - more precise and faster
3 - Perceptual Hash (pHash)

Calculate the Hash value of two images and compare the Hamming distance between them.
while the diff of Hamming distance is smaller than 5~8, two images will be considered as the same
"""

import cv2
import numpy as np


# Average Hash
def aHash(img):
    # 1 - resize to 8*8 via cubic interpolation
    img = cv2.resize(img, (8,8), interpolation = cv2.INTER_CUBIC)

    # 2 - convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # initial s as sum of pixel value, hash_str as hash value
    s = 0
    hash_str = ''

    # 3 - calculate the avg
    # iterate to get the sum of pixel at grayscale
    for i in range(8):
        for j in range(8):
            s = s + gray[i,j]
    # calculate the avg
    avg = s/64

    # 4 - compare each pixel value with the avg to generate hash str
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                hash_str = hash_str+'i'  # mark as 1 if pixel value > avg
            else:
                hash_str = hash_str+'0'  # mark as 0 if < avg
    return hash_str


# Difference Hash
def dHash(img):
    # 1- resize to 8*9
    img = cv2.resize(img, (9,8), interpolation=cv2.INTER_CUBIC)
    # 2 - convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # initialize hash_str, no need to calculate the avg of pixel value
    hash_str = ''
    # 3 - compare the current pixel value with the next pixel value to generate hash str
    for i in range(8):
        for j in range(8):
            if gray[i,j] > gray[i,j+1]:
                hash_str=hash_str+'1' # if current pixel value > the next one, mark as 1
            else:
                hash_str=hash_str+'0'
    return hash_str


# Function to compare the hash str of two images
def cmpHash(hash1, hash2):
    # initialize the Hamming distance n
    n = 0
    # set condition to make sure two hash str have the same length
    if len(hash1) != len(hash2):
        return -1  # return -1 when length not the same
    # Iterate the hash str
    for i in range(len(hash1)):
        # if the value diff, count as n+1
        if hash1[i] != hash2[i]:
            n = n+1
    return n  # return the hamming diff


# import images for comparison
img1 = cv2.imread('../images/lenna.png')
img2 = cv2.imread('../Week 4/Gaussian_noise_grayscale.png')

# use average hash to compare
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('Image similarity level via Average Hash', n)


# use difference hash to compare
hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('Image similarity level via Difference Hash', n)

