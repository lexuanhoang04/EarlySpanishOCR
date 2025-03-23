import os
from pdf2image import convert_from_path
from PIL import Image

# Fix DecompressionBombError
Image.MAX_IMAGE_PIXELS = None  

def convert_pdf_to_images(pdf_path, output_folder, dpi=300):
    """Convert Ezcaray PDF to images with no preprocessing."""
    os.makedirs(output_folder, exist_ok=True)
    print(f"Processing {pdf_path}...")

    images = convert_from_path(pdf_path, dpi=dpi)  # Keep 300 DPI for good quality
    image_paths = []

    for i, img in enumerate(images):
        img_path = os.path.join(output_folder, f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{i+1}.jpg")
        img.save(img_path, "JPEG", quality=90)  # High quality since no preprocessing
        image_paths.append(img_path)
        print(f"Saved: {img_path}")

    return image_paths

def main():
    pdf_path = "dataset/scans/Ezcaray - Vozes.pdf"  # Path to Ezcaray PDF
    output_folder = "dataset/ezcaray_images"  # Output folder for images

    convert_pdf_to_images(pdf_path, output_folder)
    print("Ezcaray processing complete!")

if __name__ == "__main__":
    main()
