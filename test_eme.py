import cv2
from QA_Metrics.EME import get_eme

img_1 = cv2.imread("ssim_0_4.png")
img_2 = cv2.imread("ssim_1_0.png")

image1_EME = get_eme(img_1)
image2_EME = get_eme(img_2)

print(image1_EME)
print(image2_EME)
