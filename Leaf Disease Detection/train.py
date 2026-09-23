"""
Leaf Disease Detection - Model Training & Evaluation Pipeline
Uses PyTorch MobileNetV2 Transfer Learning (or Custom CNN fallback).
Executes dataset loading, data augmentation, model training, evaluation metrics,
confusion matrix visualizer, loss/accuracy plots, and exports the saved model.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from generate_sample_dataset import generate_dataset, DATASET_DIR, CATEGORIES
from src.preprocessing import get_train_transforms, get_eval_transforms, IMAGENET_MEAN, IMAGENET_STD

# Configuration Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "leaf_disease_model.pt")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
METRICS_SAVE_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")
HISTORY_PLOT_PATH = os.path.join(MODELS_DIR, "training_history.png")
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")

def ensure_dataset():
    """Ensures dataset directory structure exists and contains images."""
    train_dir = os.path.join(DATASET_DIR, "train")
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        print("⚠️ Training dataset not found. Generating sample leaf dataset...")
        generate_dataset()
    else:
        print(f"📁 Dataset found at: '{DATASET_DIR}'")

class CustomCNN(nn.Module):
    """Fallback Custom CNN architecture in PyTorch."""
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def build_transfer_learning_model(num_classes):
    """Builds MobileNetV2 Transfer Learning model with custom classifier head."""
    print("🏗️ Building MobileNetV2 Transfer Learning Model (PyTorch)...")
    try:
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT
            model = models.mobilenet_v2(weights=weights)
        except AttributeError:
            model = models.mobilenet_v2(pretrained=True)

        # Unfreeze last feature blocks for fine-tuning
        for i, child in enumerate(model.features.children()):
            if i < 12:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

        # Replace classification head
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        return model
    except Exception as e:
        print(f"⚠️ Could not load MobileNetV2 weights ({e}). Falling back to Custom CNN.")
        return CustomCNN(num_classes)

def train_and_evaluate():
    """Main PyTorch training and evaluation loop."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    ensure_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Running on compute device: {device}")

    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "validation")
    test_dir = os.path.join(DATASET_DIR, "test")

    # Image Datasets & DataLoaders
    train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transforms(IMG_SIZE))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_eval_transforms(IMG_SIZE))
    test_dataset = datasets.ImageFolder(test_dir, transform=get_eval_transforms(IMG_SIZE))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = train_dataset.classes
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"🏷️ Saved {len(class_names)} class mapping labels to '{CLASS_NAMES_PATH}'.")

    model = build_transfer_learning_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # History Lists
    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []

    print(f"\n🚀 Starting PyTorch Model Training for {EPOCHS} Epochs...")

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += inputs.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += inputs.size(0)

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = correct_val / total_val

        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}%")

        # Save Best Weights checkpoint
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\n💾 Model state checkpoint saved successfully to '{MODEL_SAVE_PATH}'.")

    # --- Test Dataset Evaluation ---
    model.eval()
    all_preds = []
    all_labels = []
    test_running_loss = 0.0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_test += inputs.size(0)

    test_loss = test_running_loss / total_test
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)

    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    metrics_summary = {
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "classification_report": report_dict
    }
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"📊 Saved evaluation metrics to '{METRICS_SAVE_PATH}'.")

    # Plot & Save Accuracy/Loss History Graph
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_history, label='Training Accuracy', color='#2e7d32', linewidth=2)
    plt.plot(val_acc_history, label='Validation Accuracy', color='#1565c0', linewidth=2, linestyle='--')
    plt.title('Training vs Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(train_loss_history, label='Training Loss', color='#c62828', linewidth=2)
    plt.plot(val_loss_history, label='Validation Loss', color='#ef6c00', linewidth=2, linestyle='--')
    plt.title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=300)
    plt.close()
    print(f"📈 Training accuracy & loss plot saved to '{HISTORY_PLOT_PATH}'.")

    # Plot & Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=[c.replace('_', ' ') for c in class_names],
        yticklabels=[c.replace('_', ' ') for c in class_names]
    )
    plt.title('Confusion Matrix - Leaf Disease Classifier', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('True Label', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()
    print(f"🧩 Confusion Matrix plot saved to '{CONFUSION_MATRIX_PATH}'.")

    print("\n✨ Model Training and Evaluation Pipeline Completed Successfully!")

if __name__ == "__main__":
    train_and_evaluate()
