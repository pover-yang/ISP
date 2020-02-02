import numpy as np
import matplotlib.pyplot as plt


def read_raw_file(fp, ih, iw, dtype=np.uint32):
    """
    Read raw data from raw file
    :param fp: raw file path dir
    :param ih: raw image height
    :param iw: raw image width
    :param dtype: data type
    :return:  raw data(with numpy format)
    """
    fd = np.fromfile(fp, dtype)
    rd = fd.reshape(ih, iw)
    return rd


def show_lum_state(data):
    """
    Statistics and presentation of image luminance data
    :param data:
    :return:
    """
    plt.figure("Raw data's gray image")
    plt.imshow(data, cmap="gray")
    # plt.figure("Raw data's histogram")
    # plt.hist(data)
    plt.show()
