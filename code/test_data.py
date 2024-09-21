from scipy.io import loadmat
import cv2
import numpy as np

# all_data = loadmat("./uv/vs0000642.mat")
all_data = loadmat("./uv_distribution.mat")
# all_data = loadmat("./tb_x/tb0000027.mat")

# print(all_data["uv_distribution"].shape)
print(all_data["uv_distribution"])
# print(all_data["visibility"].shape)
# print(all_data["Y"].shape)
# print(all_data)
# a = all_data["visibility"][0, 2]
# print(a.real)
# print(a.imag)