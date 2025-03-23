import os
import cv2
import argparse
import numpy as np
from pdf2image import convert_from_path
from itertools import product

def convert_pdf_page_to_image(pdf_path, output_folder, page_num=1):
    """Convert a specific page from a PDF to an image."""
    os.makedirs(output_folder, exist_ok=True)
    
    images = convert_from_path(pdf_path, dpi=300)
    if page_num > len(images) or page_num < 1:
        print(f"Invalid page number: {page_num}. PDF has {len(images)} pages.")
        return None
    
    img = images[page_num - 1]
    img_path = os.path.join(output_folder, f"test_page_{page_num}.jpg")
    img.save(img_path, "JPEG")
    print(f"Saved test image: {img_path}")
    return img_path

def apply_clahe(img, clip_limit, tile_grid_size):
    """Enhance contrast using CLAHE with tunable parameters."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(img)

def preprocess_image(image_path, output_folder, clip_limit, tile_grid_size, block_size, C, kernel_size):
    """Apply CLAHE, thresholding, and morphological filtering with varying parameters."""
    os.makedirs(output_folder, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply CLAHE
    img_clahe = apply_clahe(img, clip_limit, tile_grid_size)

    # Apply Gaussian Blur to smooth noise before thresholding
    img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)

    # Adaptive thresholding with tunable parameters
    if block_size % 2 == 0:
        block_size += 1  # Ensure block size is odd for OpenCV
    img_bin = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, block_size, C)

    # Morphological opening with varying kernel size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_clean = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)

    # Save processed image with parameter details
    output_filename = f"clahe{clip_limit}_tile{tile_grid_size}_block{block_size}_C{C}_kernel{kernel_size}.jpg"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, img_clean)
    print(f"Generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Test different preprocessing settings")
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--output", type=str, required=True, help="Path to save generated images")
    parser.add_argument("--page", type=int, default=1, help="Page number to test (default: 1)")

    args = parser.parse_args()

    # Convert one page of the PDF to an image
    test_img_path = convert_pdf_page_to_image(args.pdf, args.output, args.page)
    if not test_img_path:
        return

    # Hyperparameter ranges
    clahe_clip_limits = [1.5, 2.0, 2.5, 3.0]  # Contrast enhancement strength
    tile_grid_sizes = [8, 16, 32]  # CLAHE tile size
    block_sizes = [15, 25, 35]  # Adaptive thresholding block size
    C_values = [5, 10, 15]  # Adaptive thresholding constant
    kernel_sizes = [2, 3, 5]  # Morphological opening kernel size

    # Generate all parameter combinations
    for clip_limit, tile_size, block_size, C, kernel_size in product(
        clahe_clip_limits, tile_grid_sizes, block_sizes, C_values, kernel_sizes
    ):
        preprocess_image(test_img_path, args.output, clip_limit, tile_size, block_size, C, kernel_size)

if __name__ == "__main__":
    main()
