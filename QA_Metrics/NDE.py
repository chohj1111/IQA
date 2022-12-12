import cv2
import numpy as np
import math

"""
T. Celik, “Two-dimensional histogram equalization and contrast enhancement,” 
Pattern Recognit., vol. 45, no. 10, pp. 3810–3824, Oct. 2012.
"""


def _get_pdf(img):
    input_img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 0]
    (H, W) = input_img_gray.shape
    num_pixel = H * W

    hist, bins = np.histogram(input_img_gray.ravel(), 256, [0, 256])

    if np.sum(hist) != num_pixel:
        raise Exception("Frequency and number of pixels do not match")

    # hist = hist.astype("object")
    # for idx, j in enumerate(hist):
    #     # print(hist[idx])
    #     hist[idx] = Fraction(hist[idx], num_pixel)
    hist = hist / num_pixel

    return hist


def _get_entropy(p):
    id_p = np.where(p != 0)
    return -np.sum(p[id_p] * np.log(p[id_p]))


def get_nde(input, target):
    input_pdf = _get_pdf(input)
    target_pdf = _get_pdf(target)

    input_entropy = _get_entropy(input_pdf)
    target_entropy = _get_entropy(target_pdf)

    nde = 1 / (1 + (math.log(256) - input_entropy) / (math.log(256) - target_entropy))
    return nde
