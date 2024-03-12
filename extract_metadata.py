import numpy

import subprocess
import numpy as np
from PIL import Image
from io import BytesIO

# Define the path to your image and exiftool
image_path = (r"C:\Users\jzieg\Kite Creator\Aluminum_Smelting - 01 Aluminum Smelting Project\04 Product Development\00 "
              r"Investigation\01 FLIR E96\FLIR8433.jpg")
exiftool_path = r"C:\Users\jzieg\Downloads\exiftool-12.77\exiftool.exe"

# Use Exiftool to extract the raw thermal image
cmd = [exiftool_path, "-RawThermalImage", "-b", image_path]
raw_thermal_image_data = subprocess.check_output(cmd)

# Convert the binary data to an image
thermal_image = Image.open(BytesIO(raw_thermal_image_data))

# Display the thermal image
thermal_image.show()

# Optional: Convert thermal image to numpy array for further analysis
thermal_array = np.array(thermal_image)
print(thermal_array.shape)
