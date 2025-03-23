import easyocr

# Initialize the EasyOCR reader
reader = easyocr.Reader(['es'])  # Use 'es' for Spanish documents

# Image path (change this to the test image)
image_path = "dataset/processed_selected/Constituciones sinodales Calahorra 1602_page_2_processed.jpg"

# Run OCR
result = reader.readtext(image_path, detail=0)

# Print extracted text
print("\n".join(result))
