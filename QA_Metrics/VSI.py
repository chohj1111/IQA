# pip install piq

import cv2
import torch
import torchvision.transforms as T
from piq import vsi

"""
L. Zhang, Y. Shen, and H. Y. Li,
“VSI: A visual saliency induced index for perceptual image quality assessment,” 
IEEE Trans. Image Process., vol. 23, no. 10, pp. 4270–4281, Oct. 2014
"""


def get_vsi(input_ndarr, target_ndarr):
    input = cv2.cvtColor(input_ndarr, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(target_ndarr, cv2.COLOR_BGR2RGB)

    transform = T.ToTensor()

    # Convert the image to PyTorch tensor
    input_tensor = transform(input).unsqueeze(0)
    target_tensor = transform(target).unsqueeze(0)

    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()

    vsi_value = vsi(input_tensor, target_tensor)

    return vsi_value.item()
