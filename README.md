# OCR Pipeline: DBNet + Transformer

This repository contains a two-stage Optical Character Recognition (OCR) pipeline:

1. Text Detection – using DBNet: https://github.com/MhLiao/DBNet
2. Text Recognition – using a Transformer-based model trained on TextOCR dataset: https://textvqa.org/textocr/

## Installation

1. Clone the repository:

   git clone https://github.com/yourusername/ocr-pipeline.git
   cd ocr-pipeline

2. Install dependencies:

   uv pip install -r requirements.txt

   Note: PyTorch must be installed separately. Please follow the official guide here:
   https://pytorch.org/get-started/locally/

3. Download pretrained weights from this link and put in the folder checkpoints

   [https://drive.google.com/drive/folders/19K3sCv3esTawo7QiO-0BHwUYbT-kGXwy?usp=sharing]

## Pipeline Overview

- Stage 1: Text Detection  
  Uses DBNet to detect word-level text regions.

- Stage 2: Text Recognition  
  A Transformer-based model recognizes text from the detected regions.

## Usage
Run this script
```bash
python src/inference_model_box2text.py\
  --json_input json/TextOCR_DB/merged_textocr.json\
  --json_output json/TextOCR/spanish_final.json\
  --config config/transformer_word_TextOCR_full.yaml\
  --image_root dataset/final_images\
  --gt_dir dataset/transcript_processed\
  --print_output
```
Note: To save time, all the materials for this script have been committed, but all the code to prepare them are available below

## Result 
- We use CER to evaluate the model, the average CER over 19 transcribed images is 0.6, which is high, but this is result of model
trained on TextOCR only, I believe CER can get lower if we finetune the model on some pages of Early Spanish dataset.

- When trained on TextOCR, the Transformer-based model got CTC Loss of 0.2 after 45 epochs and CER of 0.1 on test set of TextOCR.

## Image preprocessing
Run this script
```bash
python src/preprocess.py --pdf_folder dataset/scans --output_folder dataset/final_images
```
## Process transcription
```bash
python src/dataset/Spanish_GT_process.py\
  --input_dir dataset/transcription\
  --output_dir dataset/transcript_processed
```
dataset/transcription includes the txt (not Word) files of the transcriptions

## Process DBNet output and save in a TextOCR format JSON 
```bash
python src/other_models/db_to_textocr.py\
  --txt_dir DB\
  --image_dir DB\
  --output_dir json/TextOCR_DB\
  --merge_into_one
```
