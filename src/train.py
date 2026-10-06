import argparse
from collections import Counter
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import (
    CLASS_NAMES, DATA_DIR, IMAGE_EXTENSIONS, PROJECT_ROOT, TobaccoDataset,
    get_model, get_transforms, resolve_project_path,
)


def find_dataset_root(base_path):
    """Find a class-folder directory, including a nested archive extraction."""
    base_path = resolve_project_path(base_path)
    if not base_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {base_path}")
    queue = [base_path]
    while queue:
        current = queue.pop(0)
        subdirs = sorted(p for p in current.iterdir() if p.is_dir() and not p.name.startswith('.'))
        if any(p.name in CLASS_NAMES for p in subdirs):
            return current
        queue.extend(subdirs)
    raise ValueError(f"No Tobacco3482 class folders found under {base_path}")


def collect_samples(root_dir):
    """Require exactly ten nonempty class folders and validate image decoding."""
    root_dir = resolve_project_path(root_dir)
    found = {p.name for p in root_dir.iterdir() if p.is_dir() and not p.name.startswith('.')}
    missing = set(CLASS_NAMES) - found
    unexpected = found - set(CLASS_NAMES)
    if missing or unexpected:
        raise ValueError(f"Invalid class folders. Missing: {sorted(missing)}; unexpected: {sorted(unexpected)}")
    samples = []
    for label, name in enumerate(CLASS_NAMES):
        files = sorted(p for p in (root_dir / name).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise ValueError(f"No supported images in class folder: {root_dir / name}")
        samples.extend((str(p), label) for p in files)
    dataset = TobaccoDataset(samples)
    for i in range(len(dataset)):
        dataset[i]  # Reject unreadable images before writing any experiment files.
    return samples


def train(output_dir=None):
    output_dir = resolve_project_path(output_dir) if output_dir is not None else PROJECT_ROOT / 'training_outputs'
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Choose a new --output-dir.")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    root_dir = find_dataset_root(DATA_DIR / 'Tobacco3482')
    samples = collect_samples(root_dir)
    class_names = list(CLASS_NAMES)
    print(f"Dataset root: {root_dir} | Images: {len(samples)} | Classes: {class_names}")
    paths = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=0.1, random_state=seed, stratify=labels
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.1111, random_state=seed, stratify=train_val_labels
    )

    def stored_samples(paths, labels):
        return [(str(resolve_project_path(path).relative_to(PROJECT_ROOT).as_posix()), label) for path, label in zip(paths, labels)]

    splits = {
        'train': stored_samples(train_paths, train_labels),
        'val': stored_samples(val_paths, val_labels),
        'test': stored_samples(test_paths, test_labels),
    }

    # Create PyTorch datasets
    train_transform, val_transform = get_transforms()
    train_dataset = TobaccoDataset(splits['train'], transform=train_transform)
    val_dataset = TobaccoDataset(splits['val'], transform=val_transform)
    
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(splits['test'])}")
    
    # Loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
    
    # Initialize MobileNetV2 with pre-trained weights
    model = get_model(num_classes=len(class_names), pretrained=True)
    
    # Freeze feature parameters in blocks 0–13; BatchNorm running statistics still adapt.
    print("Freezing early layers (0 to 13) of MobileNetV2 feature extractor...")
    for i in range(14):
        for param in model.features[i].parameters():
            param.requires_grad = False
            
    model = model.to(device)
    
    # Calculate class weights for unbalanced classes
    train_labels = [s[1] for s in splits['train']]
    counter = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = len(class_names)
    class_weights = [total_samples / (num_classes * counter[i]) for i in range(num_classes)]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Calculated class weights: {class_weights}")
    
    # Loss, Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # We only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training Loop
    epochs = 12
    best_val_acc = 0.0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Create a separate experiment only after dataset validation, splitting, and model setup succeed.
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'class_mapping.json').open('w', encoding='utf-8') as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, indent=4)
    with (output_dir / 'splits.json').open('w', encoding='utf-8') as f:
        json.dump(splits, f, indent=4)
    print(f"New training artifacts will be saved to {output_dir}")
    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_weight = 0.0
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            batch_weight = class_weights_tensor[labels].sum().item()
            running_loss += loss.item() * batch_weight
            running_weight += batch_weight
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_bar.set_postfix(loss=loss.item())
            
        epoch_train_loss = running_loss / running_weight
        epoch_train_acc = correct_train / total_train
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        running_val_weight = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                batch_weight = class_weights_tensor[labels].sum().item()
                running_val_loss += loss.item() * batch_weight
                running_val_weight += batch_weight
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / running_val_weight
        epoch_val_acc = correct_val / total_val
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Step the scheduler based on validation loss
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | LR: {current_lr:.6f}")
        
        # Save best model weights
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), output_dir / 'document_classifier.pth')
            print(f"--> Saved best model weights with Val Acc: {best_val_acc:.4f}")
            
    # Save training history
    with (output_dir / 'training_history.json').open('w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)
        
    print("Training completed! Best Validation Accuracy: {:.4f}".format(best_val_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a new model without replacing the supplied checkpoint.')
    parser.add_argument('--output-dir', default='training_outputs', help='New, empty project-relative output directory.')
    train(parser.parse_args().output_dir)
