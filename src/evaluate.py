import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from utils import (
    DATA_DIR, MODEL_DIR, PROJECT_ROOT, REPORT_DIR, TobaccoDataset,
    get_model, get_transforms, load_class_names, resolve_project_path,
)

def evaluate(run_dir=None, output_dir=None):
    model_dir = resolve_project_path(run_dir) if run_dir is not None else MODEL_DIR
    split_path = model_dir / 'splits.json' if run_dir is not None else DATA_DIR / 'splits.json'
    output_dir = resolve_project_path(output_dir) if output_dir is not None else PROJECT_ROOT / 'evaluation_outputs'
    if output_dir.resolve() == REPORT_DIR.resolve():
        raise ValueError('Use a separate output directory to preserve the supplied reports.')
    class_names = load_class_names(model_dir / 'class_mapping.json')
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Missing split manifest: {split_path}. Restore the original splits for the supplied checkpoint, "
            "or evaluate a new training run with --run-dir. Do not invent a new test split for existing weights."
        )
    with split_path.open(encoding='utf-8') as f:
        splits = json.load(f)
    test_split = splits.get('test', [])
    if not test_split:
        raise ValueError('The test split is empty.')
    if any(not isinstance(label, int) or not 0 <= label < len(class_names) for _, label in test_split):
        raise ValueError('Test labels must be integer class indices from 0 to 9.')
    if {label for _, label in test_split} != set(range(len(class_names))):
        raise ValueError('The test split must include all ten classes for per-class evaluation.')
    test_paths = [resolve_project_path(path).resolve() for path, _ in test_split]
    other_paths = {resolve_project_path(path).resolve() for key in ('train', 'val') for path, _ in splits.get(key, [])}
    if len(set(test_paths)) != len(test_paths) or set(test_paths) & other_paths:
        raise ValueError('Test paths are duplicated or overlap the training/validation splits.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for evaluation: {device} | Test images: {len(test_split)}")
    _, val_transform = get_transforms()
    test_dataset = TobaccoDataset(test_split, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    model = get_model(num_classes=len(class_names), pretrained=False)
    checkpoint_path = model_dir / 'document_classifier.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True), strict=True)
    model = model.to(device)
    model.eval()

    # Inference
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Running evaluation on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 1. Classification report
    report_dict = classification_report(all_labels, all_preds, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(all_labels, all_preds, labels=list(range(len(class_names))), target_names=class_names, zero_division=0)
    print("\nClassification Report:\n", report_text)
    
    # 2. Confusion matrix
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Confusion Matrix - Tobacco3482 Classifier')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
    
    # 3. Precision-Recall Curve (for each class)
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        y_true_binary = (all_labels == i).astype(int)
        y_scores = all_probs[:, i]
        
        precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
        ap = average_precision_score(y_true_binary, y_scores)
        
        plt.plot(recall, precision, label=f'{class_name} (AP = {ap:.4f})')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Tobacco3482 Classifier')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall.png', dpi=300)
    plt.close()
    print(f"Saved precision-recall curves to {output_dir / 'precision_recall.png'}")
    
    # 4. Generate evaluation_report.md
    history_md = ""
    if (model_dir / 'training_history.json').is_file():
        with (model_dir / 'training_history.json').open(encoding='utf-8') as f:
            history = json.load(f)
        
        # Plot training curves and save
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='Train Loss')
        plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Val Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(history['train_acc']) + 1), history['train_acc'], label='Train Accuracy')
        plt.plot(range(1, len(history['val_acc']) + 1), history['val_acc'], label='Val Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=300)
        plt.close()
        print(f"Saved training curves to {output_dir / 'training_curves.png'}")
        
        # Format training history in markdown
        history_md = f"""
## 📈 Training History

Here is a summary of the model training over epochs:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| :---: | :---: | :---: | :---: | :---: |
"""
        for epoch in range(len(history['train_loss'])):
            history_md += f"| {epoch+1} | {history['train_loss'][epoch]:.4f} | {history['train_acc'][epoch]:.4f} | {history['val_loss'][epoch]:.4f} | {history['val_acc'][epoch]:.4f} |\n"
            
        history_md += "\n![Training Curves](training_curves.png)\n"

    report_md = f"""# 📊 Tobacco3482 Document Classifier Evaluation Report

This report presents the metrics and charts evaluating the trained MobileNetV2 document classifier.

## 🏆 Summary Metrics

- **Test Accuracy**: {report_dict['accuracy']:.4f}
- **Macro Average F1-score**: {report_dict['macro avg']['f1-score']:.4f}
- **Weighted Average F1-score**: {report_dict['weighted avg']['f1-score']:.4f}

---

{history_md}

---

## 📋 Detailed Classification Report

Below is the classification report showing Precision, Recall, and F1-score for each of the 10 classes.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
"""
    for class_name in class_names:
        c_metrics = report_dict[class_name]
        report_md += f"| **{class_name}** | {c_metrics['precision']:.4f} | {c_metrics['recall']:.4f} | {c_metrics['f1-score']:.4f} | {int(c_metrics['support'])} |\n"
        
    report_md += f"| | | | | |\n"
    report_md += f"| **Accuracy** | | | {report_dict['accuracy']:.4f} | {int(report_dict['macro avg']['support'])} |\n"
    report_md += f"| **Macro Avg** | {report_dict['macro avg']['precision']:.4f} | {report_dict['macro avg']['recall']:.4f} | {report_dict['macro avg']['f1-score']:.4f} | {int(report_dict['macro avg']['support'])} |\n"
    report_md += f"| **Weighted Avg** | {report_dict['weighted avg']['precision']:.4f} | {report_dict['weighted avg']['recall']:.4f} | {report_dict['weighted avg']['f1-score']:.4f} | {int(report_dict['weighted avg']['support'])} |\n"

    report_md += """
---

## 🗺️ Confusion Matrix

The confusion matrix shows the true classes versus predicted classes. This helps analyze where the model confuses documents (for example, Letters vs. Memos).

![Confusion Matrix](confusion_matrix.png)

---

## 📈 Precision-Recall Curves

Precision-Recall curves measure the tradeoff between precision and recall for different thresholds. Average precision (AP) summarizes precision across recall increments; higher values indicate better ranking for that category.

![Precision-Recall Curves](precision_recall.png)
"""

    with (output_dir / 'evaluation_report.md').open('w', encoding='utf-8') as f:
        f.write(report_md)
        
    print(f"Saved evaluation report to {output_dir / 'evaluation_report.md'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate an existing checkpoint using its corresponding test split.')
    parser.add_argument('--run-dir', help='New training output directory; omit for the supplied model.')
    parser.add_argument('--output-dir', default='evaluation_outputs', help='Directory for newly generated reports.')
    args = parser.parse_args()
    evaluate(args.run_dir, args.output_dir)
