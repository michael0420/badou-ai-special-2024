import cv2
from matplotlib import pyplot as plt


def show_bgr_histogram(image):
    """
    show the histogram plot in bgr channels of the image

    :param image: the image to display plot
    :return: return nothing but show histogram plot
    """
    channels = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    for (i, channel) in enumerate(channels):
        result = cv2.calcHist(channels, [i], None, [256], [0, 256])
        plt.plot(result, color=colors[i])
        plt.xlim([0, 256])

    plt.show()


if __name__ == "__main__":
    before = cv2.imread("alex.jpg")
    (b, g, r) = cv2.split(before)
    bHist = cv2.equalizeHist(b)
    gHist = cv2.equalizeHist(g)
    rHist = cv2.equalizeHist(r)

    after = cv2.merge((bHist, gHist, rHist))
    cv2.imwrite('result/equalize_histogram.jpg', after)

    show_bgr_histogram(before)
    show_bgr_histogram(after)
