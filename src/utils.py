import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / 'models'
REPORT_DIR = PROJECT_ROOT / 'reports'
DATA_DIR = PROJECT_ROOT / 'data'
CLASS_NAMES = ('ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific')
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def load_class_names(mapping_path=None):
    """Validate the output order before using a checkpoint."""
    mapping_path = Path(mapping_path) if mapping_path is not None else MODEL_DIR / 'class_mapping.json'
    with mapping_path.open(encoding='utf-8') as f:
        mapping = json.load(f)
    expected = {str(i): name for i, name in enumerate(CLASS_NAMES)}
    if mapping != expected:
        raise ValueError('Class mapping must contain the ten expected classes in checkpoint output order.')
    return list(CLASS_NAMES)


def resolve_project_path(path):
    """Resolve project-relative paths, including Windows separators in split files."""
    path = Path(str(path).replace('\\', '/'))
    return path if path.is_absolute() else PROJECT_ROOT / path


class TobaccoDataset(Dataset):
    """
    A PyTorch Dataset wrapper for local image paths.
    Converts PIL images to RGB and applies torchvision transforms.
    """
    def __init__(self, samples, transform=None):
        """
        samples: list of tuples (image_path, label_idx)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_path = resolve_project_path(img_path)
        
        try:
            with Image.open(img_path) as source:
                image = source.convert('RGB')
                image.load()
        except Exception as e:
            raise RuntimeError(f"Cannot read dataset image '{img_path}': {e}") from e
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_model(num_classes=10, pretrained=True):
    """
    Returns a MobileNetV2 model with the final classification layer
    adjusted for the number of target classes.
    """
    if pretrained:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    else:
        model = models.mobilenet_v2(weights=None)
        
    # Replace the final linear layer in the classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def get_transforms():
    """
    Returns data transforms for training and validation/testing.
    """
    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for generating visual explanations.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []
        
        # Register forward and backward hooks
        self.handlers.append(self.target_layer.register_forward_hook(self.save_activation))
        self.handlers.append(self.target_layer.register_full_backward_hook(self.save_gradient))
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def __call__(self, input_tensor, target_category=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_category is None:
            target_category = output.argmax(dim=1).item()
            
        loss = output[0, target_category]
        loss.backward()
        
        activations = self.activations
        gradients = self.gradients
        
        if activations is None or gradients is None:
            raise RuntimeError("Activations or gradients are None. Verify target layer hook registration.")
            
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3))[0]
        
        # Weighted combination of activation maps
        heatmap = torch.zeros(activations.shape[2:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            heatmap += w * activations[0, i, :, :]
            
        # Apply ReLU
        heatmap = torch.clamp(heatmap, min=0)
        
        # Normalize heatmap
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
            
        return heatmap.cpu().numpy(), target_category
        
    def remove_hooks(self):
        for h in self.handlers:
            h.remove()

def generate_gradcam_overlay(image_pil, heatmap_arr, alpha=0.45):
    """
    Resizes heatmap, maps it to a color palette, and overlays it on the original image.
    """
    # 1. Resize heatmap to match original image size
    img_w, img_h = image_pil.size
    heatmap_resized = Image.fromarray((heatmap_arr * 255).astype(np.uint8))
    heatmap_resized = heatmap_resized.resize((img_w, img_h), Image.Resampling.BILINEAR)
    heatmap_resized_arr = np.array(heatmap_resized) / 255.0
    
    # 2. Apply colormap
    cm = plt.get_cmap('jet')
    heatmap_colored = cm(heatmap_resized_arr)[:, :, :3]  # scale [0, 1] RGB
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # 3. Blend original image with heatmap
    img_arr = np.array(image_pil.convert('RGB'))
    blended = (1.0 - alpha) * img_arr + alpha * heatmap_colored
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended), Image.fromarray(heatmap_colored)
