# EarlySpanishOCR
OCR Pipeline: DBNet + Transformer
This repository contains a two-stage Optical Character Recognition (OCR) pipeline:

Text Detection – using DBNet.

Text Recognition – using a Transformer-based model.

📦 Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/ocr-pipeline.git
cd ocr-pipeline
Install dependencies:

bash
Copy
Edit
uv pip freeze > requirements.txt  # or use the provided one
uv pip install -r requirements.txt
Note: PyTorch must be installed separately. Follow the official PyTorch installation guide to install the correct version for your system.

Download pretrained weights:

Model Weights Download Link

🧠 Pipeline Overview
Stage 1: Text Detection

Utilizes DBNet to detect word-level text regions.

Stage 2: Text Recognition

Applies a Transformer-based architecture to recognize text from cropped regions.

📄 Usage
Coming soon.
