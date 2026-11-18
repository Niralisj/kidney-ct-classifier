# 🩻 Kidney CT Scan Classifier with Explainable AI

A deep learning model that classifies kidney CT slices as **Normal, Cyst, Tumor or Stone**,
explains each prediction with **Grad-CAM** heatmaps, and flags low-confidence scans for
radiologist review. It comes with an interactive **Streamlit** app.

## Results (held-out test set, 1,790 images)

| Model | Test accuracy |
|---|---|
| Baseline CNN (trained from scratch) | 51.7% |
| **EfficientNetB0 (transfer learning + fine-tuning)** | **88.6%** |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Cyst | 0.890 | 0.982 | 0.933 |
| Normal | 0.980 | 0.847 | 0.909 |
| Stone | 0.620 | 0.897 | 0.733 |
| Tumor | 0.937 | 0.827 | 0.879 |

**Uncertainty flag (confidence < 80%):**
- 25.9% of scans are flagged for radiologist review
- Accuracy on confident predictions: **96.8%** (vs 65.3% on uncertain ones)
- The flag catches **40 of the 59** tumors the model would otherwise miss

![Confusion matrix](C:\Users\NIRALI\Downloads\kidney-ct-classifier\images\confusion_matrix.png)
![Grad-CAM examples](C:\Users\NIRALI\Downloads\kidney-ct-classifier\images\gradcam_examples.png)

## Approach

1. **Data cleaning and leakage check.** Hashing every image revealed 120 test images that were
   exact copies of training images. The dataset was deduplicated *before* splitting
   (12,446 → 11,929 unique images; 517 duplicates removed, 0 label conflicts).
2. **Stratified 70 / 15 / 15 split**, so every split keeps the same class proportions.
3. **Augmentation** (flip, rotation, zoom) on training images only.
4. **Class weights** to handle imbalance (Stone is the minority class).
5. **Baseline CNN** trained from scratch as a benchmark.
6. **EfficientNetB0** pretrained on ImageNet: the head was trained with the backbone frozen, then the
   last 30 layers were fine-tuned at a learning rate of 1e-5, with BatchNorm kept frozen and
   EarlyStopping to prevent overfitting.
7. **Grad-CAM** on the `top_activation` layer for visual explanations.
8. **Confidence threshold** that defers uncertain cases to a human.

## Key findings

- **Transfer learning** lifted test accuracy from 51.7% to 88.6%.
- **Duplicate leakage** is a likely reason many public notebooks on this dataset report 99%+ accuracy.
- **Grad-CAM** shows the model often attends to the kidney region, but sometimes to the spine,
  which suggests possible **shortcut learning**.
- **Most missed tumors** were still flagged as another abnormality; only 12 of 342 (3.5%) were
  predicted as Normal.

## Limitations and future work

- The dataset has no patient IDs, so a patient-level split was not possible; results may still be optimistic.
- Single 2D slices, not full 3D scans or clinical context.
- Next steps: crop to the kidney region (or segment it first) to reduce shortcut learning, raise tumor
  recall (class-weight tuning, a larger backbone), and validate on data from another hospital.
- **Educational project. Not a medical device.**

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Sample scans to try are in `demo_images/`.

## Project structure

```
├── app.py                  # Streamlit app
├── gradcam.py              # Grad-CAM utilities
├── kidney_ct_model.keras   # trained model
├── notebooks/              # training notebook (Colab)
├── images/                 # confusion matrix and Grad-CAM examples
├── demo_images/            # sample CT slices
└── requirements.txt
```

## Dataset

[CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) (Kaggle, MD Nazmul Islam), CC BY 4.0.

## Tech stack

Python · TensorFlow/Keras · EfficientNetB0 · scikit-learn · Grad-CAM · Streamlit · Google Colab (T4 GPU)
