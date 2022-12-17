import cv2
import QA.QA_Metrics as QA


ori_frame = cv2.imread("data/Part2/blurnoise/market.bmp")
blurred_frame = cv2.imread("data/Part2/blurnoise/market_blur1_noise2.bmp")

# EME
print("### EME ###")
print(QA.get_eme(ori_frame))
print(QA.get_eme(blurred_frame))

# BRISQUE
print("### BRISQUE ###")
print(QA.get_brisque(ori_frame))
print(QA.get_brisque(blurred_frame))

# NIQE
print("### NIQE ###")
print(QA.get_niqe(ori_frame))
print(QA.get_niqe(blurred_frame))


# NDE
# print(QA.get_nde(ori_frame, enhanced_frame))

# SSIM
print("### SSIM ###")
print(QA.get_ssim(ori_frame, blurred_frame))

# VSI
print("### VSI ###")
print(QA.get_vsi(ori_frame, blurred_frame))
