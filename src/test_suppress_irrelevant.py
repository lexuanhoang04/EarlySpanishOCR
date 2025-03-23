import os
import cv2
import argparse
import numpy as np
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
    """Enhance contrast using CLAHE with best found parameters."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(img)

def preprocess_image(image_path, output_folder, clip_limit=1.5, tile_grid_size=8, kernel_size=3):
    """Apply preprocessing that suppresses irrelevant text while keeping main text clear."""
    os.makedirs(output_folder, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply CLAHE
    img_clahe = apply_clahe(img, clip_limit, tile_grid_size)

    # Apply Stronger Gaussian Blur
    img_blur = cv2.GaussianBlur(img_clahe, (9, 9), 0)  # Increased from (7,7) to (9,9)

    # Use Otsu's Thresholding (better for high contrast backgrounds)
    _, img_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing followed by opening
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_closed = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)  # Fill in gaps
    img_clean = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel)  # Remove specks

    # Save processed image
    output_filename = os.path.basename(image_path).replace(".jpg", "_suppress_test.jpg")
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, img_clean)
    print(f"Generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Test preprocessing to suppress irrelevant text on *Constituciones sinodales*")
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed image")
    parser.add_argument("--page", type=int, default=1, help="Page number to test (default: 1)")

    args = parser.parse_args()

    # Convert one page of the PDF to an image
    test_img_path = convert_pdf_page_to_image(args.pdf, args.output, args.page)
    if test_img_path:
        # Apply preprocessing
        preprocess_image(test_img_path, args.output)

if __name__ == "__main__":
    main()
