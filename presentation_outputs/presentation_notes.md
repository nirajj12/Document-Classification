# 10-Minute Presentation Outline

Use these points to prepare slides and a short demo. The suggested times total ten minutes. Describe saved results as supplied historical artifacts; runtime checks do not establish classification accuracy.

## 1. Project Introduction — 30 seconds

- Automated Document Classifier: predict a scanned image's document category.
- Python, PyTorch, MobileNetV2, Grad-CAM, Streamlit.
- Input: document image. Output: class, softmax scores, explanatory overlay.

## 2. Problem — 30 seconds

- Scanned documents need category labels for organization and retrieval.
- Explore image-based classification without a separate OCR stage.
- No comparison experiment establishes superiority over OCR.

## 3. Dataset — 45 seconds

- Tobacco3482: 3,482 benchmark images, ten categories.
- ADVE (Advertisement), Email, Form, Letter, Memo, News, Note, Report, Resume, Scientific.
- Small, imbalanced dataset; some category labels are ambiguous.
- Dataset and original splits are absent locally. Upload inference does not need them.

## 4. Preprocessing — 40 seconds

- RGB conversion → resize to 384×384 → tensor → ImageNet normalization.
- Tensor shape for one image: [1, 3, 384, 384].
- Training adds horizontal flips and rotations up to ±10°.
- Validation, test, and deployed inference share deterministic transforms.

## 5. MobileNetV2 — 45 seconds

- CNN feature extractor using depthwise separable convolutions and inverted residual blocks.
- Final feature map: [1, 1280, 12, 12] at the chosen input resolution.
- Global average pooling → 1280 features → Dropout(0.2) → Linear(1280, 10).
- Ten raw logits; softmax produces ten class scores.

## 6. Transfer Learning — 35 seconds

- ImageNet weights initialize training rather than starting every parameter randomly.
- Feature parameters in blocks 0–13 frozen; 14–18 and classifier trainable.
- BatchNorm running statistics still adapt in early blocks.
- Application loads the full saved checkpoint without downloading ImageNet weights.

## 7. Training — 50 seconds

- Approximately 80/10/10 stratified train/validation/test split.
- Weighted CrossEntropyLoss uses inverse class counts from training only.
- Adam, learning rate 0.0001, batch size 32, twelve epochs.
- ReduceLROnPlateau monitors validation loss; best validation accuracy selects checkpoint.
- Show training_curves.png. Explain the train/validation difference cautiously.
- No training was conducted during cleanup. Future runs use a separate output folder.

## 8. Evaluation — 55 seconds

- Accuracy: total correct fraction.
- Precision: correctness among predictions of a class. Recall: fraction of that class found.
- F1 balances precision and recall; macro treats classes equally, weighted uses supports.
- Confusion matrix rows are actual labels; columns are predicted labels.
- Precision-recall curves use one-versus-rest scores; AP summarizes ranking performance.

## 9. Grad-CAM — 55 seconds

- Target: final MobileNetV2 feature block.
- Capture activations and gradients of the selected class logit.
- Spatially average gradients → weight channels → sum → ReLU → normalize.
- Resize 12×12 heatmap and blend with image.
- Warm regions indicate larger positive class contributions within that map.
- Approximate explanation; no proof that the model understands text or semantics.

## 10. Streamlit Application — 75 seconds

- Start inside the document-classifier environment: streamlit run src/app.py.
- Upload a real document image prepared before the presentation.
- Show predicted class and top-two softmax scores.
- Switch Original / Grad-CAM / Compare; open Model Performance and About.
- Explain actual CPU/CUDA selection and model caching.
- A synthetic-image smoke test verifies execution only, never accuracy.
- Use historical screenshots as backup, noting their older UI wording.

## 11. Results — 55 seconds

- Supplied test accuracy: 85.10% = 297/349.
- Macro F1: 84.13%; weighted F1: 85.09%.
- Final training accuracy: 86.78%; validation accuracy: 79.94%.
- Strongest saved F1: ADVE 0.9787, News 0.9730, Email 0.9672.
- Weakest: Scientific 0.5769; Form recall 0.7209.
- Scientific → Letter/Memo/Report: three cases each. Do not invent the cause.
- Original splits are missing, so these are not newly reproduced results.

## 12. Limitations — 45 seconds

- Ten classes only; no unknown-document rejection.
- Softmax score is not calibrated confidence.
- Domain shift, small/imbalanced data, label quality, and potentially related pages across splits.
- Square resizing and approximate Grad-CAM.
- Image uploads only; TIFF first page; no PDF or OCR processing.

## 13. Future Scope — 40 seconds

- More representative data and improved document augmentation.
- Duplicate/group-aware splits and clearer experiment records.
- Unknown-class detection and confidence calibration.
- Compare other CNNs in a controlled experiment.

## Output-to-Slide Guide

| File | What it shows / why use it | Slide or topic |
|---|---|---|
| training_curves.png | Training/validation loss and accuracy over twelve epochs | Training and generalization |
| confusion_matrix.png | Actual versus predicted category counts; inspect errors | Evaluation and error analysis |
| precision_recall.png | Per-class precision/recall curves and AP | Evaluation metrics |
| ../reports/evaluation_report.md | Summary, per-class precision/recall/F1, supports | Results; reference for questions |
| ../models/training_history.json | Numerical history underlying the curves | Training; source-code explanation |
| dashboard.png | Historical dashboard and upload workflow | Application overview / demo backup |
| prediction_gradcam.png | Historical prediction, class scores, original image and Grad-CAM together | Inference and explainability / demo backup |
| results_summary.md | One-page implementation/results reminder | Presenter reference |
