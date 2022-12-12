from skimage.metrics import structural_similarity
import numpy as np

def get_ssim(input, target):
    input_min = np.iinfo(input.dtype).min
    input_max = np.iinfo(input.dtype).max

    ssim_value, _ = structural_similarity(target, input, data_range=input_max - input_min, channel_axis=2, full=True)

    return ssim_value
