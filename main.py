# main.py

from analysis.thermal_processor import ThermalProcessor
from analysis.utilities import save_thermal_data, save_image


def main():
    image_path = "data/FLIR8433.jpg"
    processor = ThermalProcessor(image_path)
    processor.process_image()
    processor.display_thermal_image()

    # Example: Save the thermal data
    if processor.thermal_np is not None:
        save_thermal_data(processor.thermal_np, "results/thermal_data.csv")
        save_image(processor.thermal_np, "results/thermal_image.png")

    # Add additional analysis steps as needed


if __name__ == "__main__":
    main()
