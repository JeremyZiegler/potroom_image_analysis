# analysis/thermal_processor.py

import numpy as np
from flirimageextractor import FlirImageExtractor
import matplotlib.pyplot as plt


class ThermalProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.fir = FlirImageExtractor()
        self.thermal_np = None

    def process_image(self):
        self.fir.process_image(self.image_path)
        self.thermal_np = self.fir.get_thermal_np()

    def display_thermal_image(self):
        if self.thermal_np is not None:
            plt.imshow(self.thermal_np, cmap='hot')
            plt.colorbar()
            plt.title("Thermal Image")
            plt.show()
        else:
            print("Thermal data not available. Please process the image first.")

    # Add more methods for analysis here (ROI, thresholding, histogram, etc.)
