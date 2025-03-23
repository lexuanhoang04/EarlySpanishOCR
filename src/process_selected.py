import os
import cv2
import argparse
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

# Fix DecompressionBombError
Image.MAX_IMAGE_PIXELS = None  

def convert_pdf_to_images(pdf_path, output_folder):
    """Convert all pages of a PDF into images."""
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=300)

    image_paths = []
    for i, img in enumerate(images):
        img_path = os.path.join(output_folder, f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{i+1}.jpg")
        img.save(img_path, "JPEG")
        image_paths.append(img_path)
        print(f"Saved: {img_path}")

    return image_paths

def apply_clahe(img, clip_limit=1.5, tile_grid_size=8):
    """Enhance contrast using CLAHE with best settings."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(img)

def preprocess_image(image_path, output_folder, clip_limit=1.5, tile_grid_size=8, block_size=15, C=10, kernel_size=2):
    """Apply CLAHE, thresholding, and morphological filtering with best found settings."""
    os.makedirs(output_folder, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply CLAHE
    img_clahe = apply_clahe(img, clip_limit, tile_grid_size)

    # Apply Gaussian Blur to smooth noise before thresholding
    img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)

    # Adaptive thresholding
    img_bin = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, block_size, C)

    # Morphological opening
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_clean = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)

    # Save processed image
    output_filename = os.path.basename(image_path).replace(".jpg", "_processed.jpg")
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, img_clean)
    print(f"Generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process *Constituciones sinodales* and *Mendo* with best preprocessing settings")
    parser.add_argument("--pdf_folder", type=str, required=True, help="Folder containing the selected PDFs")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save processed images")

    args = parser.parse_args()

    pdf_files = [
        "Constituciones sinodales Calahorra 1602.pdf",
        "Mendo - Principe perfecto.pdf"
    ]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(args.pdf_folder, pdf_file)
        image_paths = convert_pdf_to_images(pdf_path, args.output_folder)

        for img_path in image_paths:
            preprocess_image(img_path, args.output_folder)

if __name__ == "__main__":
    main()
