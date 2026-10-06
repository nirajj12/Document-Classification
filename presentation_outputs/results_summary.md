# Results Summary

## System

- **Task:** Document image classification without a separate OCR stage.
- **Dataset:** Tobacco3482, reported benchmark size 3,482 images in ten classes. Dataset files are not supplied locally.
- **Classes:** ADVE (Advertisement), Email, Form, Letter, Memo, News, Note, Report, Resume, Scientific.
- **Model:** MobileNetV2, global average pooling, Dropout(0.2), Linear(1280, 10).
- **Input:** RGB, resized to 384×384, tensor conversion, ImageNet normalization.
- **Training code:** ImageNet initialization; feature parameters 0–13 frozen, 14–18 trainable; early BatchNorm statistics still adapt. Weighted CrossEntropyLoss, Adam at 0.0001, batch size 32, twelve epochs, ReduceLROnPlateau, approximately 80/10/10 stratified split. Best validation accuracy selects the checkpoint.

## Supplied Results

These values come from saved artifacts, **not a newly reproduced training or test evaluation**. The matrix and report are internally consistent; the original split manifest is missing.

| Metric | Saved value |
|---|---:|
| Test accuracy | **85.10%** — 297 correct / 349 test images |
| Macro F1 | **84.13%** |
| Weighted F1 | **85.09%** |
| Final training accuracy | 86.78% |
| Final validation accuracy | 79.94% |

**Strongest F1:** ADVE 0.9787, News 0.9730, Email 0.9672.

**Weakest:** Scientific, precision/recall/F1 all 0.5769, with 26 test images. Its saved AP is 0.6787. Form recall is 0.7209. Scientific is misclassified as Letter, Memo, and Report three times each; the cause is not established by the matrix.

## Presentation Evidence

- [Training curves](training_curves.png): learning over twelve epochs; the historical x-axis uses indices 0–11.
- [Confusion matrix](confusion_matrix.png): correct predictions and category confusions.
- [Precision-recall curves](precision_recall.png): class ranking performance and saved AP.
- [Dashboard](dashboard.png) and [prediction with Grad-CAM](prediction_gradcam.png): historical interface examples; older wording differs from the cleaned app.
- [Full evaluation report](../reports/evaluation_report.md) and [training history](../models/training_history.json): detailed metrics and numerical history.

## Key Limitations

Every input is forced into a known class; softmax scores are not calibrated confidence. New document styles may reduce accuracy. The dataset is small and imbalanced, with known label ambiguities. Square resizing distorts aspect ratio. Grad-CAM is coarse and approximate, without proving semantic understanding. Original test results cannot be independently reproduced without matching images and splits; image-level splitting does not detect duplicate or related pages.
