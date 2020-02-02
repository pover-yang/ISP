import cv2
import numpy as np


def demosaic(bayer_data):
    """
    bayer to RGB
    :param bayer_data:
    :return:
    """
    hei, wid = bayer_data.shape
    rgb_data = np.zeros(shape=(hei, wid, 3), dtype=bayer_data.dtype)
    paded_data = np.pad(bayer_data, 1, 'reflect')
    for h in range(hei):
        for w in range(wid):
            if h % 2 == 0 and w % 2 == 0:  # R Pixel
                rgb_data[h, w, 0] = paded_data[h + 1, w + 1]  # r
                rgb_data[h, w, 1] = round((paded_data[h, w + 1] + paded_data[h + 1, w] +
                                           paded_data[h + 1, w + 2] + paded_data[h + 2, w + 1]) / 4)  # g
                rgb_data[h, w, 2] = round((paded_data[h, w] + paded_data[h, w + 2] +
                                           paded_data[h + 2, w] + paded_data[h + 2, w + 2]) / 4)  # b
            elif h % 2 == 1 and w % 2 == 1:  # B Pixel
                rgb_data[h, w, 0] = round((paded_data[h, w] + paded_data[h, w + 2] +
                                           paded_data[h + 2, w] + paded_data[h + 2, w + 2]) / 4)  # r
                rgb_data[h, w, 1] = round((paded_data[h, w + 1] + paded_data[h + 1, w] +
                                           paded_data[h + 1, w + 2] + paded_data[h + 2, w + 1]) / 4)  # g
                rgb_data[h, w, 2] = paded_data[h + 1, w + 1]  # b
            elif h % 2 == 0 and w % 2 == 1:  # Gr Pixel
                rgb_data[h, w, 0] = round((paded_data[h + 1, w] + paded_data[h + 1, w + 2]) / 2)  # r
                rgb_data[h, w, 1] = paded_data[h + 1, w + 1]  # g
                rgb_data[h, w, 2] = round((paded_data[h, w + 1] + paded_data[h + 2, w + 1]) / 2)  # b
            else:  # Gb Pixel
                rgb_data[h, w, 0] = round((paded_data[h, w + 1] + paded_data[h + 2, w + 1]) / 2)  # r
                rgb_data[h, w, 1] = paded_data[h + 1, w + 1]  # g
                rgb_data[h, w, 2] = round((paded_data[h + 1, w] + paded_data[h + 1, w + 2]) / 2)  # b
    return rgb_data


def demosaic_float(bayer_data):
    """
    bayer to RGB
    :param bayer_data:
    :return:
    """
    hei, wid = bayer_data.shape
    rgb_data = np.zeros(shape=(hei, wid, 3), dtype=bayer_data.dtype)
    paded_data = np.pad(bayer_data, 1, 'reflect')
    for h in range(hei):
        for w in range(wid):
            if h % 2 == 0 and w % 2 == 0:  # R Pixel
                rgb_data[h, w, 0] = paded_data[h + 1, w + 1]  # r
                rgb_data[h, w, 1] = (paded_data[h, w + 1] + paded_data[h + 1, w] +
                                     paded_data[h + 1, w + 2] + paded_data[h + 2, w + 1]) / 4  # g
                rgb_data[h, w, 2] = (paded_data[h, w] + paded_data[h, w + 2] +
                                     paded_data[h + 2, w] + paded_data[h + 2, w + 2]) / 4  # b
            elif h % 2 == 1 and w % 2 == 1:  # B Pixel
                rgb_data[h, w, 0] = (paded_data[h, w] + paded_data[h, w + 2] +
                                     paded_data[h + 2, w] + paded_data[h + 2, w + 2]) / 4  # r
                rgb_data[h, w, 1] = (paded_data[h, w + 1] + paded_data[h + 1, w] +
                                     paded_data[h + 1, w + 2] + paded_data[h + 2, w + 1]) / 4  # g
                rgb_data[h, w, 2] = paded_data[h + 1, w + 1]  # b
            elif h % 2 == 0 and w % 2 == 1:  # Gr Pixel
                rgb_data[h, w, 0] = (paded_data[h + 1, w] + paded_data[h + 1, w + 2]) / 2  # r
                rgb_data[h, w, 1] = paded_data[h + 1, w + 1]  # g
                rgb_data[h, w, 2] = (paded_data[h, w + 1] + paded_data[h + 2, w + 1]) / 2  # b
            else:  # Gb Pixel
                rgb_data[h, w, 0] = (paded_data[h, w + 1] + paded_data[h + 2, w + 1]) / 2  # r
                rgb_data[h, w, 1] = paded_data[h + 1, w + 1]  # g
                rgb_data[h, w, 2] = (paded_data[h + 1, w] + paded_data[h + 1, w + 2]) / 2  # b
    return rgb_data


def rgb2gray(rgb_data):
    """
    RGB to gray(luminance value only)
    :param rgb_data:
    :return:
    """
    gray_data = np.round((306 * rgb_data[:, :, 0] + 601 * rgb_data[:, :, 1] + 117 * rgb_data[:, :, 2]) / 1024)
    return gray_data.astype(rgb_data.dtype)


def bayer2y(bayer_data):
    """
    bayer to gray(luminance value only)
    :param bayer_data:
    :return:
    """
    hei, wid = bayer_data.shape
    rgb_data = np.zeros(shape=(hei, wid, 3), dtype=bayer_data.dtype)
    gray_data = np.zeros(shape=bayer_data.shape, dtype=bayer_data.dtype)
    paded_data = np.pad(bayer_data, 1, 'reflect')
    for h in range(hei):
        for w in range(wid):
            if h % 2 == 0 and w % 2 == 0:  # R Pixel
                rgb_data[h, w, 0] = paded_data[h + 1, w + 1]  # r
                rgb_data[h, w, 1] = round((paded_data[h, w + 1] + paded_data[h + 1, w] +
                                           paded_data[h + 1, w + 2] + paded_data[h + 2, w + 1]) / 4)  # g
                rgb_data[h, w, 2] = round((paded_data[h, w] + paded_data[h, w + 2] +
                                           paded_data[h + 2, w] + paded_data[h + 2, w + 2]) / 4)  # b
            elif h % 2 == 1 and w % 2 == 1:  # B Pixel
                rgb_data[h, w, 0] = round((paded_data[h, w] + paded_data[h, w + 2] +
                                           paded_data[h + 2, w] + paded_data[h + 2, w + 2]) / 4)  # r
                rgb_data[h, w, 1] = round((paded_data[h, w + 1] + paded_data[h + 1, w] +
                                           paded_data[h + 1, w + 2] + paded_data[h + 2, w + 1]) / 4)  # g
                rgb_data[h, w, 2] = paded_data[h + 1, w + 1]  # b
            elif h % 2 == 0 and w % 2 == 1:  # Gr Pixel
                rgb_data[h, w, 0] = round((paded_data[h + 1, w] + paded_data[h + 1, w + 2]) / 2)  # r
                rgb_data[h, w, 1] = paded_data[h + 1, w + 1]  # g
                rgb_data[h, w, 2] = round((paded_data[h, w + 1] + paded_data[h + 2, w + 1]) / 2)  # b
            else:  # Gb Pixel
                rgb_data[h, w, 0] = round((paded_data[h, w + 1] + paded_data[h + 2, w + 1]) / 2)  # r
                rgb_data[h, w, 1] = paded_data[h + 1, w + 1]  # g
                rgb_data[h, w, 2] = round((paded_data[h + 1, w] + paded_data[h + 1, w + 2]) / 2)  # b
            gray_data[h, w] = round(
                (306 * rgb_data[h, w, 0] + 601 * rgb_data[h, w, 1] + 117 * rgb_data[h, w, 2]) / 1024)
    return np.clip(gray_data, 0, pow(2, 20) - 1)
