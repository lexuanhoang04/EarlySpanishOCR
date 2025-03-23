import os
import cv2
import argparse
import numpy as np
from itertools import product
from pdf2image import convert_from_path
from PIL import Image

# Fix DecompressionBombError
Image.MAX_IMAGE_PIXELS = None  

def convert_pdf_page_to_image(pdf_path, output_folder, page_num=1):
    """Convert a specific page from a PDF to an image."""
    os.makedirs(output_folder, exist_ok=True)
    
    images = convert_from_path(pdf_path, dpi=300)
    if page_num > len(images) or page_num < 1:
        print(f"Invalid page number: {page_num}. PDF has {len(images)} pages.")
        return None
    
    img = images[page_num - 1]
    img_path = os.path.join(output_folder, f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{page_num}.jpg")
    img.save(img_path, "JPEG")
    print(f"Saved test image: {img_path}")
    return img_path

def apply_clahe(img, clip_limit=1.5, tile_grid_size=8):
    """Enhance contrast using CLAHE with tunable parameters."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(img)

def preprocess_image(image_path, output_folder, blur_size, tile_grid_size, thresholding, block_size, C, kernel_size):
    """Apply preprocessing with different configurations."""
    os.makedirs(output_folder, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply CLAHE
    img_clahe = apply_clahe(img, 1.5, tile_grid_size)

    # Apply Gaussian Blur
    img_blur = cv2.GaussianBlur(img_clahe, (blur_size, blur_size), 0)

    # Apply Thresholding
    if thresholding == "adaptive":
        img_bin = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, block_size, C)
    else:  # Otsu’s Thresholding
        _, img_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological Transform
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_closed = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)  # Fill gaps
    img_clean = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel)  # Remove specks

    # Save processed image
    output_filename = f"blur{blur_size}_tile{tile_grid_size}_thr{thresholding}_block{block_size}_C{C}_kernel{kernel_size}.jpg"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, img_clean)
    print(f"Generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Test multiple preprocessing settings on page 2 of *Constituciones sinodales*")
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed images")
    parser.add_argument("--page", type=int, default=2, help="Page number to test (default: 2)")

    args = parser.parse_args()

    # Convert one page of the PDF to an image
    test_img_path = convert_pdf_page_to_image(args.pdf, args.output, args.page)
    if not test_img_path:
        return

    # Define hyperparameter ranges
    blur_sizes = [3, 5, 7, 9]  # Different Gaussian blur strengths
    tile_grid_sizes = [8, 16]  # CLAHE tile sizes
    thresholding_methods = ["adaptive", "otsu"]
    block_sizes = [15, 21, 25]  # Adaptive thresholding block size
    C_values = [5, 10, 15]  # Adaptive thresholding constant
    kernel_sizes = [3, 5]  # Morphological kernel sizes

    # Generate all parameter combinations
    for blur, tile, thr, block, C, kernel in product(blur_sizes, tile_grid_sizes, thresholding_methods, block_sizes, C_values, kernel_sizes):
        preprocess_image(test_img_path, args.output, blur, tile, thr, block, C, kernel)

if __name__ == "__main__":
    main()
