import os
import argparse
from pdf2image import convert_from_path
import cv2
import numpy as np
from glob import glob

def convert_pdf_to_images(pdf_path, output_folder):
    """Convert a multi-page PDF into images."""
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=300)
    
    image_paths = []
    for i, img in enumerate(images):
        img_path = os.path.join(output_folder, f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{i+1}.jpg")
        img.save(img_path, "JPEG")
        image_paths.append(img_path)
        print(f"Saved: {img_path}")
    
    return image_paths

def preprocess_image(image_path, output_folder):
    """Apply grayscale, binarization, and denoising to an image."""
    os.makedirs(output_folder, exist_ok=True)
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply adaptive thresholding for binarization
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Optional: Remove noise with morphological transformations
    kernel = np.ones((1, 1), np.uint8)
    img_clean = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
    
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img_clean)
    print(f"Processed: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess PDF scans for OCR")
    parser.add_argument("--pdf_folder", type=str, required=True, help="Path to folder containing PDFs")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save processed images")

    args = parser.parse_args()

    # Convert all PDFs in the folder to images
    pdf_files = glob(os.path.join(args.pdf_folder, "*.pdf"))
    for pdf_file in pdf_files:
        image_paths = convert_pdf_to_images(pdf_file, args.output_folder)

        # Preprocess each extracted image
        for img_path in image_paths:
            preprocess_image(img_path, args.output_folder)

if __name__ == "__main__":
    main()
