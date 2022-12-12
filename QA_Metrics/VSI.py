# pip install piq

import cv2
import torchvision.transforms as T
from piq import vsi


def get_vsi(input_ndarr, target_ndarr):
    input = cv2.cvtColor(input_ndarr, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(target_ndarr, cv2.COLOR_BGR2RGB)

    transform = T.ToTensor()

    # Convert the image to PyTorch tensor
    input_tensor = transform(input).unsqueeze(0)
    target_tensor = transform(target).unsqueeze(0)

    vsi_value = vsi(input_tensor, target_tensor)

    return vsi_value.item()
