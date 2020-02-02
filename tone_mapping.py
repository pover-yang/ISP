import numpy as np
from math import exp, floor
from cv2 import bilateralFilter
lum_white = (pow(2, 20) - 1)


def tone_mapping(lum_world):
    lum_log_mean, lum_log_min, lum_log_max, diff_ratio, lum_max = data_statistic(lum_world)
    lum_world -= np.min(lum_world)
    lum_mean = pow(2, float(lum_log_mean))
    theta = 6309.94 * pow(0.62, float(lum_log_mean)) + 4.17
    key = lum_mean * theta
    lw_scale = 1 + key/lum_white
    bf_lum_world = bilateral_filter_cv(lum_world, color_sigma=2, space_sigma=1, win_size=5, data_width=20)
    # bf_lum_world = bilateral_filter(lum_world, color_sigma=2, space_sigma=1, win_size=5)
    lum_dst = lum_world / (key * pow(bf_lum_world/(pow(2, 20)-1), 0.5) + lum_world) * lw_scale
    return lum_dst.astype(np.float32)


def data_statistic(lum):
    log2_gray_data = np.log2(lum + 1)
    lum_log_mean = np.mean(log2_gray_data)
    lum_log_min = np.percentile(log2_gray_data, 0.04, interpolation='midpoint')
    lum_log_max = np.percentile(log2_gray_data, 99.96, interpolation='midpoint')
    diff_ratio = (2 * lum_log_mean - lum_log_max - lum_log_min) / (lum_log_max - lum_log_min)
    lum_max = np.max(lum)
    return lum_log_mean, lum_log_min, lum_log_max, diff_ratio, lum_max


def bilateral_filter(src_data, color_sigma, space_sigma, win_size, data_width):
    operator_data = (src_data/(pow(2, data_width-8)-1)).astype(np.int8)
    dst_data = np.zeros(src_data.shape, src_data.dtype)
    color_coeff = -1 / (2 * color_sigma ** 2)
    space_coeff = -1 / (2 * space_sigma ** 2)
    radius = win_size // 2
    space_weight_win = np.zeros((win_size, win_size))
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            space_weight_win[i + radius, j + radius] = \
                np.floor(exp((i ** 2 + j ** 2) * space_coeff) * (pow(2, data_width-8)-1))
    # calculate color weight win and bilateral weight win
    paded_data = np.pad(operator_data, 2, 'reflect')
    hei, wid = src_data.shape[:2]
    for h in range(hei):
        for w in range(wid):
            data_win = paded_data[h + 2 - radius:h + 3 + radius, w + 2 - radius:w + 3 + radius]
            color_diff_win = (data_win - data_win[radius, radius]) / 4095
            color_weight_win = np.exp(np.power(color_diff_win, 2) * color_coeff)
            bilateral_weight_win = color_weight_win * space_weight_win * pow(4095, 2)
            dst_data[h, w] = np.sum(bilateral_weight_win * data_win) / np.sum(bilateral_weight_win)
    return dst_data


def bilateral_filter_cv(src_data, color_sigma, space_sigma, win_size, data_width):
    src_data.dtype = "float32"
    dst_data = bilateralFilter(src=src_data, d=win_size, sigmaColor=color_sigma, sigmaSpace=space_sigma)
    src_data.dtype = "int32"
    dst_data.dtype = "int32"
    dst_data /= (pow(2, data_width) - 1)
    return dst_data
