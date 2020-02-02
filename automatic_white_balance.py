from math import floor
import numpy as np
RED_GAIN = 700 / 431
GREEN_GAIN = 1
BLUE_GAIN = 600 / 412


def bayer_awb_float(bayer_raw, data_width, r_g=RED_GAIN, g_g=GREEN_GAIN, b_g=BLUE_GAIN):
    """
    automatic white balance with rgb gain value
    :param bayer_raw: bayer raw data
    :param data_width: input data width for clip
    :param r_g: red gain
    :param g_g: green gain
    :param b_g: blue gain
    :return: AWB processed data
    """
    awb_rd = np.zeros(shape=bayer_raw.shape, dtype=bayer_raw.dtype)
    hei, wid = bayer_raw.shape
    for h in range(hei):
        for w in range(wid):
            if h % 2 == 0 and w % 2 == 0:
                awb_rd[h, w] = bayer_raw[h, w] * r_g
            elif h % 2 == 1 and w % 2 == 1:
                awb_rd[h, w] = bayer_raw[h, w] * b_g
            else:
                awb_rd[h, w] = bayer_raw[h, w] * g_g
    max_val = np.array(pow(2, data_width)-1, np.int32)
    max_val.dtype = "float32"
    return np.clip(awb_rd, 0, max_val)


def bayer_awb_int(bayer_raw, data_width, r_g=RED_GAIN, g_g=GREEN_GAIN, b_g=BLUE_GAIN):
    """
    automatic white balance with rgb gain value
    :param bayer_raw: bayer raw data
    :param data_width: input data width for clip
    :param r_g: red gain
    :param g_g: green gain
    :param b_g: blue gain
    :return: AWB processed data
    """
    r_g = floor(r_g * 1024)  # use Q10
    g_g = floor(g_g * 1024)
    b_g = floor(b_g * 1024)
    awb_rd = np.zeros(shape=bayer_raw.shape, dtype=bayer_raw.dtype)
    hei, wid = bayer_raw.shape
    for h in range(hei):
        for w in range(wid):
            if h % 2 == 0 and w % 2 == 0:
                awb_rd[h, w] = floor(bayer_raw[h, w] * r_g / 1024)
            elif h % 2 == 1 and w % 2 == 1:
                awb_rd[h, w] = floor(bayer_raw[h, w] * b_g / 1024)
            else:
                awb_rd[h, w] = floor(bayer_raw[h, w] * g_g / 1024)
    return np.clip(awb_rd, 0, pow(2, data_width)-1)
