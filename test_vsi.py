import cv2
from QA_Metrics.VSI import get_vsi

input_img = cv2.imread("ssim_0_4.png")
target_img = cv2.imread("ssim_1_0.png")

print(get_vsi(input_img, target_img))
