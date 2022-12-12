import math

import cv2
import numpy as np

"""
S. S. Agaian, B. Silver, and K. A. Panetta, 
“Transform coefficient histogram-based image enhancement algorithms using contrast entropy,” 
IEEE Trans. Image Process., vol. 16, no. 3, pp. 741–758, Mar. 2007.
"""


def get_eme(img):
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lumi, Cr, Cb = cv2.split(ycrcb_img)

    w = 8
    # window size for EME
    (R, C) = lumi.shape
    tmp_sum = 0.0

    for j in range(0, math.floor(R / w)):
        for i in range(0, math.floor(C / w)):
            J = j * w
            I = i * w

            tmp_block = lumi[J : J + w, I : I + w].copy()

            block_max = np.max(tmp_block)
            block_min = np.min(tmp_block)

            if block_max == block_min:
                continue
            else:
                tmp_sum = tmp_sum + 20.0 * math.log(block_max / (block_min + 1e-4))

    value = (tmp_sum / (R * C)) * (w**2)

    return value
