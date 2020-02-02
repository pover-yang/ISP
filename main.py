import cv2
import numpy as np
import matplotlib.pyplot as plt
from common_methods import *
from color_conversion import *
from tone_mapping import tone_mapping
from automatic_white_balance import *
INPUT_DW = 20
raw_data_max = 2 ** INPUT_DW - 1
RAW_FILE_PATH = "D:/MyProjects/Luminance/data/raw/IMX490_20bit_1440_928_day/camera0_raw_result_000.raw"
IMG_HEI = 928
IMG_WID = 1440

if __name__ == "__main__":
    # todo :Open windows to select target file
    # bayer_raw_float = read_raw_file(RAW_FILE_PATH, IMG_HEI, IMG_WID, dtype=np.float32)
    # bayer_awb = bayer_awb_float(bayer_raw_float, INPUT_DW)
    bayer_raw_int = read_raw_file(RAW_FILE_PATH, IMG_HEI, IMG_WID, dtype=np.int32)
    bayer_awb = bayer_awb_int(bayer_raw_int, INPUT_DW)
    # print("HDR Image MaxVal = " + str(raw_awb_data.max()), "\nHDR Image MinVal = " + str(raw_awb_data.min()))
    luminance_world = bayer2y(bayer_awb)
    # show_lum_state(luminance_world)
    luminance_dst = tone_mapping(luminance_world)
    bayer_dst = pow((bayer_awb / (luminance_world + 1)), 0.4) * luminance_dst
    rgb_dst = demosaic_float(bayer_dst)
    plt.imshow(rgb_dst)
