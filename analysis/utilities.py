# analysis/utilities.py

import numpy as np
import matplotlib.pyplot as plt


def save_thermal_data(thermal_data, file_name):
    np.savetxt(file_name, thermal_data, delimiter=",")


def save_image(data, file_name, cmap='hot'):
    plt.imsave(file_name, data, cmap=cmap)
