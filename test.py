import cv2
from QA_Metrics.EME import get_eme
from QA_Metrics.NDE import get_nde
from QA_Metrics.SSIM import get_ssim
from QA_Metrics.VSI import get_vsi


ori_frame = cv2.imread("extracted_0000.tiff")
enhanced_frame = cv2.imread("0000.tiff")

print(get_eme(ori_frame))
print(get_eme(enhanced_frame))

# NDE
print(get_nde(ori_frame, enhanced_frame))

# SSIM
print(get_ssim(ori_frame, enhanced_frame))

# VSI
print(get_vsi(ori_frame, enhanced_frame))
