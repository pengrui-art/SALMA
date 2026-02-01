import os
from PIL import Image


def set_image_dpi(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(directory, filename)
            try:
                img = Image.open(filepath)
                # Check if DPI is already 300
                if "dpi" in img.info:
                    current_dpi = img.info["dpi"]
                    if current_dpi[0] == 300 and current_dpi[1] == 300:
                        print(f"Skipping {filename}, already 300 DPI")
                        continue

                print(f"Processing {filename}...")

                # Save with 300 DPI
                # For PNG, dpi parameter is enough.
                # For JPEG, same.

                # We need to save to a temp file then replace or overwrite
                # Overwriting directly works usually

                img.save(filepath, dpi=(300, 300))
                print(f"Set {filename} to 300 DPI")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")


if __name__ == "__main__":
    figures_dir = os.path.join(os.path.dirname(__file__), "figures")
    if os.path.exists(figures_dir):
        set_image_dpi(figures_dir)
    else:
        print(f"Directory not found: {figures_dir}")
