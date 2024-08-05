# -*- coding:utf-8 -*-

__author__ = 'Young'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def image_kmeans(image, k, attempts):
    compactness, labels, centers = __kmeans(image.copy(), k, attempts)
    dst = __dst(centers, image, labels)
    return __build_image(dst, image, labels)


def __kmeans(image, k, attempts):
    k_data = __k_data(image)
    stop_criteria = __stop_criteria()
    flags = cv2.KMEANS_RANDOM_CENTERS
    return cv2.kmeans(k_data, k, None, stop_criteria, attempts, flags)


def __k_data(image):
    k_shape = lambda i: (image.shape[0] * image.shape[1], 1) if len(i.shape) == 2 else (-1, 3)
    k_data = image.reshape(k_shape(image))
    return np.float32(k_data)


def __stop_criteria():
    return cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0


def __dst(centers, image, labels):
    is_gray = len(image.shape) == 2
    img_shape = image.shape
    if is_gray:
        return labels.reshape(img_shape[0], img_shape[1])
    else:
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        return res.reshape(img_shape)


def __build_image(dst, image, labels):
    is_gray = len(image.shape) == 2
    if (is_gray):
        return labels.reshape(image.shape)
    else:
        return cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)


def show_kmeans_results(titles, images):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    num = len(images)
    for i in range(num):
        plt.subplot(num // 2, (num + 2) // 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def kmeans_athlete(X, k):
    clf = KMeans(n_clusters=k)
    predict = clf.fit_predict(X)
    return predict


def show_kmeans_predict(X, predict):
    x = [n[0] for n in X]
    y = [n[1] for n in X]
    plt.scatter(x, y, c=predict, marker='x')
    plt.title("Kmeans-Basketball Data")
    plt.xlabel("assists_per_minute")
    plt.ylabel("points_per_minute")
    plt.legend(["A", "B", "C"])
    plt.show()
