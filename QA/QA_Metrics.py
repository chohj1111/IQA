import math

import cv2
import torch
import torchvision.transforms as T
import numpy as np
import pyiqa

from skimage.metrics import structural_similarity


"""
    No Reference
"""


def get_brisque(img):
    """
    Mittal, Anish, Anush Krishna Moorthy, and Alan Conrad Bovik.
    "No-reference image quality assessment in the spatial domain."
    IEEE Transactions on image processing 21.12 (2012): 4695-4708.
    """
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    iqa_metric = pyiqa.create_metric("brisque", device=dev)

    transform = T.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(dev)

    return iqa_metric(input_tensor).item()


def get_eme(img):
    """
    S. S. Agaian, B. Silver, and K. A. Panetta,
    “Transform coefficient histogram-based image enhancement algorithms using contrast entropy,”
    IEEE Trans. Image Process., vol. 16, no. 3, pp. 741–758, Mar. 2007."""
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


def get_niqe(img):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    iqa_metric = pyiqa.create_metric("niqe", test_y_channel=True, color_space="ycbcr", device=dev)

    transform = T.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(dev)

    return iqa_metric(input_tensor).item()


"""
    Full Reference
"""


def get_ssim(input, target):
    """
    Wang, Zhou, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli,
    "Image quality assessment: from error visibility to structural similarity,"
    IEEE Trans. Image Process., vol. 13, no. 4, pp. 600-612, Apr. 2004"""

    input_min = np.iinfo(input.dtype).min
    input_max = np.iinfo(input.dtype).max

    ssim_value, _ = structural_similarity(target, input, data_range=input_max - input_min, channel_axis=2, full=True)

    return ssim_value


def get_vsi(input_img, target_img):
    """
    L. Zhang, Y. Shen, and H. Y. Li,
    “VSI: A visual saliency induced index for perceptual image quality assessment,”
    IEEE Trans. Image Process., vol. 23, no. 10, pp. 4270–4281, Oct. 2014"""
    input = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

    # Convert the image to PyTorch tensor
    transform = T.ToTensor()
    input_tensor = transform(input).unsqueeze(0)
    target_tensor = transform(target).unsqueeze(0)

    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()

    vsi_value = vsi(input_tensor, target_tensor)

    return vsi_value.item()


def get_vsi_pyiqa(input_img, target_img):
    """
    Mittal, Anish, Anush Krishna Moorthy, and Alan Conrad Bovik.
    "No-reference image quality assessment in the spatial domain."
    IEEE Transactions on image processing 21.12 (2012): 4695-4708.
    """
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    iqa_metric = pyiqa.create_metric("vsi", device=dev)

    transform = T.ToTensor()
    input_tensor = transform(input_img).unsqueeze(0).to(dev)
    target_tensor = transform(target_img).unsqueeze(0).to(dev)

    return iqa_metric(input_tensor, target_tensor).item()


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
"""
