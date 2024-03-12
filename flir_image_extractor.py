import cv2
import numpy as np
import matplotlib.pyplot as plt
from flirimageextractor import FlirImageExtractor


class ThermalImageAnalyzer:
    def __init__(self, image_path, threshold):
        """
        Initializes the thermal image analyzer with the given path and threshold.

        Parameters:
        - image_path: str, path to the thermal image file
        - threshold: float, temperature threshold to detect anomalies
        """
        self.image_path = image_path
        self.threshold = threshold
        self.fir = FlirImageExtractor()
        self.thermal_np = None
        self.anomalies = None
        self.contours = None

    def load_and_process_image(self):
        """Loads and processes the thermal image to extract the thermal data."""
        self.fir.process_image(self.image_path)
        self.thermal_np = self.fir.get_thermal_np()

    def detect_anomalies(self):
        """
        Detects anomalies in the thermal image based on the defined threshold.
        Anomalies are regions with temperatures above the threshold.
        """
        self.anomalies = self.thermal_np > self.threshold
        self.contours, _ = cv2.findContours(self.anomalies.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def display_thermal_image(self):
        """Displays the thermal image with a color map representing temperature."""
        plt.imshow(self.thermal_np, cmap='hot')
        plt.colorbar()
        plt.title("Thermal Image")
        plt.show()

    def display_anomalies(self):
        """Highlights the anomalies on the thermal image and displays it."""
        plt.imshow(self.thermal_np, cmap='hot', interpolation='nearest')
        plt.colorbar()

        # Highlight the anomalies with rectangles
        for i, contour in enumerate(self.contours):
            # Find the bounding box of each contour
            plt.plot(contour[:, 0, 0], contour[:, 0, 1], color='blue', linewidth=2)
            # Draw the rectangle

        # Add a summary of the count of anomalies
        plt.text(0.95, 0.95, f'Anomalies detected: {len(self.contours)}',
                 verticalalignment='top', horizontalalignment='right',
                 color='white', fontsize=12, transform=plt.gca().transAxes)

        plt.title(f"Anomalies in Thermal Image (Threshold: {self.threshold}°C)")
        plt.show()

    def get_user_defined_roi(self):
        """
        Prompt the user to define an ROI by clicking the bottom left and top right corners on the image.
        """
        if not hasattr(self, 'rois'):
            self.rois = {}

        # Define the event handler for mouse click
        def onclick(event):
            # If it is the first click, store the bottom left corner
            if not hasattr(onclick, 'bottom_left'):
                onclick.bottom_left = (int(event.xdata), int(event.ydata))
                print(f"Bottom left corner selected at: {onclick.bottom_left}")
            # If it is the second click, store the top right corner and display the ROI
            else:
                top_right = (int(event.xdata), int(event.ydata))
                print(f"Top right corner selected at: {top_right}")

                # Calculate the width and height of the ROI
                width = top_right[0] - onclick.bottom_left[0]
                height = top_right[1] - onclick.bottom_left[1]

                # Get the name of the ROI from the user
                roi_name = input("Enter the name of this ROI: ")

                # Define the ROI with the user-provided name and calculated dimensions
                self.define_roi(roi_name, onclick.bottom_left, width, height)

                # Disconnect the event handler after the second click
                plt.disconnect(binding_id)
                plt.close()  # Close the figure window

        # Show the image to the user
        fig, ax = plt.subplots()
        ax.imshow(self.thermal_np, cmap='hot')
        plt.colorbar()
        plt.title("Click the bottom left and top right corners of the ROI")

        # Bind the onclick event handler to the figure
        binding_id = plt.connect('button_press_event', onclick)
        plt.show()

    def define_roi(self, roi_name, top_left, width, height):
        """
        Define a Region of Interest (ROI) within the thermal image.

        Parameters:
        - roi_name: str, a unique name for the ROI
        - top_left: tuple, the (x, y) coordinates of the top-left corner of the ROI
        - width: int, the width of the ROI
        - height: int, the height of the ROI
        """
        # Store the ROI as a dictionary entry, each with a name and associated rectangle coordinates
        if not hasattr(self, 'rois'):
            self.rois = {}
        self.rois[roi_name] = (top_left, top_left[0] + width, top_left[1] + height)

    def check_rois_for_anomalies(self):
        """
        Check each defined ROI for anomalies and highlight them in the thermal image.
        """
        # Check if ROIs have been defined
        if not hasattr(self, 'rois'):
            print("No ROIs defined.")
            return

        for roi_name, (top_left, bottom_right_x, bottom_right_y) in self.rois.items():
            # Extract the ROI from the anomalies image
            roi_anomalies = self.anomalies[top_left[1]:bottom_right_y, top_left[0]:bottom_right_x]

            # Check if there are any anomalies in the ROI
            if np.any(roi_anomalies):
                print(f"Anomaly detected in {roi_name}.")

                # Find contours for anomalies within this ROI
                contours, _ = cv2.findContours(roi_anomalies.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

                # Adjust contours to match the position within the whole image
                contours = [contour + np.array([top_left[0], top_left[1]]) for contour in contours]

                # Highlight the contours on the image
                cv2.drawContours(self.thermal_np, contours, -1, (255, 0, 0), 2)

            else:
                print(f"No anomalies detected in {roi_name}.")


# Example usage
image_path = r"C:\Users\jzieg\Kite Creator\Aluminum_Smelting - 01 Aluminum Smelting Project\04 Product Development\00 "\
             r"Investigation\01 FLIR E96\FLIR8433.jpg"
threshold = 45
#
# analyzer = ThermalImageAnalyzer(image_path, threshold)
# analyzer.load_and_process_image()
# analyzer.detect_anomalies()
# analyzer.display_thermal_image()  # Optionally, display the original thermal image
# analyzer.display_anomalies()  # Display anomalies
# Example usage
# Initialize the analyzer
analyzer = ThermalImageAnalyzer(image_path, threshold)

# Load and process the image
analyzer.load_and_process_image()

# Interactively get ROI from user
analyzer.get_user_defined_roi()

# Detect anomalies
analyzer.detect_anomalies()

# Check the ROIs for anomalies
analyzer.check_rois_for_anomalies()

# Display the thermal image with highlighted anomalies
analyzer.display_anomalies()
