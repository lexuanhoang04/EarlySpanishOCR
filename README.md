# OCR Pipeline: DBNet + Transformer

This repository contains a two-stage Optical Character Recognition (OCR) pipeline:

1. Text Detection – using DBNet: https://github.com/MhLiao/DBNet
2. Text Recognition – using a Transformer-based model

## Installation

1. Clone the repository:

   git clone https://github.com/yourusername/ocr-pipeline.git
   cd ocr-pipeline

2. Install dependencies:

   uv pip install -r requirements.txt

   Note: PyTorch must be installed separately. Please follow the official guide here:
   https://pytorch.org/get-started/locally/

3. Download pretrained weights from this link:

   [insert your weights link here]

## Pipeline Overview

- Stage 1: Text Detection  
  Uses DBNet to detect word-level text regions.

- Stage 2: Text Recognition  
  A Transformer-based model recognizes text from the detected regions.

## Usage

Coming soon.
