import numpy as np
from skimage.metrics import structural_similarity

"""
Wang, Zhou, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli,
"Image quality assessment: from error visibility to structural similarity," 
IEEE Trans. Image Process., vol. 13, no. 4, pp. 600-612, Apr. 2004
"""


def get_ssim(input, target):
    input_min = np.iinfo(input.dtype).min
    input_max = np.iinfo(input.dtype).max

    ssim_value, _ = structural_similarity(
        target, input, data_range=input_max - input_min, channel_axis=2, full=True
    )

    return ssim_value
