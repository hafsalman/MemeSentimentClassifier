import os
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import pytesseract
import torch
from torchvision import models, transforms
from transformers import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import shap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_labels(csv_path):
    return pd.read_csv(csv_path)

def run_ocr_on_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return ""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return pytesseract.image_to_string(gray).strip()

def extract_resnet_features(image_path, model, preprocess, device):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model(x)
    return feats.cpu().numpy().reshape(-1)

def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "gradcam")
    ensure_dir(out_dir / "shap")
    images_dir = data_dir / "images"
    labels_csv = data_dir / "labels.csv"

    labels_df = load_labels(labels_csv)

    texts = []
    rows = []
    for _, row in labels_df.iterrows():
        fn = row['filename']
        img_path = images_dir / fn
        text = run_ocr_on_image(img_path)
        texts.append(text)
        rows.append({'filename': fn, 'text': text, 'label': row['label']})

    extracted_df = pd.DataFrame(rows)
    extracted_df.to_csv(out_dir / "extracted_texts.csv", index=False)

    text_pipe = pipeline("sentiment-analysis")
    text_scores = []
    for t in extracted_df['text'].fillna("").tolist():
        if t.strip() == "":
            text_scores.append({'label': 'NEUTRAL', 'score': 0.5})
        else:
            text_scores.append(text_pipe(t[:512])[0])

    numeric_text_feats = []
    for o in text_scores:
        lab = o['label'].upper()
        sc = float(o['score'])
        if lab == "POSITIVE":
            numeric_text_feats.append([sc, 0.0])
        elif lab == "NEGATIVE":
            numeric_text_feats.append([0.0, sc])
        else:
            numeric_text_feats.append([0.0, 0.0])

    text_feat_arr = np.array(numeric_text_feats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet50(pretrained=True)
    resnet.eval()
    resnet.to(device)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    img_feats = []
    for fn in extracted_df['filename']:
        feats = extract_resnet_features(images_dir / fn, resnet, preprocess, device)
        img_feats.append(feats)

    img_feats = np.vstack(img_feats)

    X = np.hstack([img_feats, text_feat_arr])
    le = LabelEncoder()
    y = le.fit_transform(extracted_df['label'].values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Validation accuracy: {acc:.3f}")

    probs = clf.predict_proba(X_scaled)
    preds = clf.predict(X_scaled)
    out_rows = []
    for i, fn in enumerate(extracted_df['filename']):
        out_rows.append({
            'filename': fn,
            'text': extracted_df.loc[i,'text'],
            'true_label': extracted_df.loc[i,'label'],
            'pred_label': le.inverse_transform([preds[i]])[0],
            'prob_positive': float(probs[i].max())
        })
    pd.DataFrame(out_rows).to_csv(out_dir / "predictions.csv", index=False)

    X_text = text_feat_arr
    clf_text = LogisticRegression(max_iter=500)
    try:
        clf_text.fit(X_text, y)
        explainer = shap.LinearExplainer(clf_text, X_text, feature_dependence="independent")
        shap_vals = explainer.shap_values(X_text)
        for i in range(len(X_text)):
            html = f"<html><body><h3>{extracted_df.loc[i,'filename']}</h3>"
            html += f"<p>Extracted text: <b>{extracted_df.loc[i,'text']}</b></p><ul>"
            html += f"<li>pos_score: {X_text[i,0]:.3f}, SHAP: {shap_vals[0][i,0]:.4f}</li>"
            html += f"<li>neg_score: {X_text[i,1]:.3f}, SHAP: {shap_vals[0][i,1]:.4f}</li>"
            html += f"</ul><p>Predicted label: <b>{out_rows[i]['pred_label']}</b></p></body></html>"
            with open(out_dir / "shap" / f"shap_text_{i}.html", "w", encoding="utf-8") as f:
                f.write(html)
    except Exception as e:
        print("SHAP text explanation failed:", e)

    target_layers = [resnet.layer4[-1]]
    cam = GradCAM(model=resnet, target_layers=target_layers)

    for i in idx_test[:5]:
        fn = extracted_df.loc[i,'filename']
        img_path = images_dir / fn
        img = Image.open(img_path).convert("RGB")

        img_resized = img.resize((224, 224))
        rgb = np.array(img_resized).astype(np.float32) / 255.0

        input_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = resnet(input_tensor)
            imgnet_pred = out.argmax(dim=1).item()

        targets = [ClassifierOutputTarget(imgnet_pred)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        visualization = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
        Image.fromarray(visualization).save(out_dir / "gradcam" / f"gradcam_{i}_{fn}")
        print("Saved", out_dir / "gradcam" / f"gradcam_{i}_{fn}")

    print("Pipeline completed. Check the output folder for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing images/ and labels.csv")
    parser.add_argument("--out_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(args)