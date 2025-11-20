# Meme Sentiment Classifier - Quick Demo
This repository contains a minimal end-to-end pipeline to demo a **Meme Sentiment Classifier** (text + image) with simple explainability (SHAP for text, Grad-CAM for images).

## Structure
- `demo_pipeline.py` - single script that runs OCR, extracts text, uses a text sentiment pipeline, extracts image features, trains a light fusion classifier, and produces simple explanations.
- `requirements.txt` - Python packages to install.
- `sample_labels.csv` - example CSV format for dataset labeling (you must provide images in `data/images/` folder).
- `run_demo.sh` - helper bash script to run the pipeline (after installing dependencies).

## Quick setup
1. Create project folder and move the images to `data/images/`.
2. Create `data/labels.csv` with columns: `filename,label` (labels: positive, negative, offensive).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install Tesseract OCR on your system:
   - Ubuntu: `sudo apt install tesseract-ocr`
   - Mac: `brew install tesseract`
   - Windows: download installer from https://github.com/tesseract-ocr/tesseract

5. Run the demo:
   ```bash
   python demo_pipeline.py --data_dir data --out_dir output --epochs 3
   ```

Notes:
- This is a minimal demo intended to produce a working pipeline quickly. For production you should fine-tune models, add validation, and improve preprocessing.
