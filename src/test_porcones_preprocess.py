import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

# Fix DecompressionBombError
Image.MAX_IMAGE_PIXELS = None  

def convert_pdf_to_images(pdf_path, output_folder, dpi=100, max_width=1500, max_height=2000):
    """Convert all pages of a PDF into images with decompression handling."""
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi)  # Lower DPI to 100

    image_paths = []
    for i, img in enumerate(images):
        # Convert to grayscale
        img = img.convert("L")

        # Resize to limit memory usage
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)  # Fixed ANTIALIAS issue

        img_path = os.path.join(output_folder, f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{i+1}.jpg")
        img.save(img_path, "JPEG", quality=85)  # Reduce quality slightly for smaller size
        image_paths.append(img_path)
        print(f"Saved: {img_path}")

    return image_paths

def main():
    pdf_path = "dataset/scans/Porcones.pdf"  # Path to Porcones
    output_folder = "dataset/test_porcones_preprocessed"  # Test output folder

    print("Testing Porcones preprocessing...")
    convert_pdf_to_images(pdf_path, output_folder)
    print("Test completed! Check images in:", output_folder)

if __name__ == "__main__":
    main()
