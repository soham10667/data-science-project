# Leaf Disease Detection – Image-Based Plant Health Inspection System 🌿

An end-to-end Data Science and Deep Learning application that automatically analyzes plant leaf images to classify them as **Healthy** or **Infected**, identifies specific plant diseases, computes prediction confidence, and provides actionable treatment and prevention guidelines via an interactive Streamlit web application.

---

## 🎯 Project Objective

Plant diseases pose a severe threat to agricultural yields and food security. Early and accurate detection allows farmers and agronomists to apply targeted treatments before infections spread. 

This project demonstrates computer vision, data preprocessing, deep learning transfer learning (MobileNetV2), model evaluation (confusion matrix, precision/recall/F1-score), and a user-friendly Streamlit web interface for real-time inspection.

---

## ✨ Key Features

1. **Leaf Image Upload & Validation**:
   - Supports JPG, JPEG, and PNG image formats.
   - Validates file integrity, RGB channel format, and resolution before processing.
   - Provides live preview and image dimension metadata.

2. **Image Preprocessing & Data Augmentation**:
   - Resizes input images to standard $224 \times 224$ pixels.
   - Normalizes pixel intensity values to $[0, 1]$.
   - Training pipeline applies real-time data augmentations: rotation ($\pm 25^\circ$), horizontal flip, zoom, shear, and width/height shifts.

3. **Deep Learning Model (Transfer Learning)**:
   - Built with TensorFlow/Keras using **MobileNetV2** pre-trained on ImageNet (with Custom CNN fallback option).
   - GlobalAveragePooling2D, Batch Normalization, Dropout (0.3), and Softmax Output layer for multi-class categorization.

4. **Automated Training & Evaluation Pipeline**:
   - Split dataset into **Training**, **Validation**, and **Testing** sets.
   - Callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`.
   - Generates evaluation graphs: **Training vs Validation Loss & Accuracy** and **Confusion Matrix Heatmap**.
   - Saves model to `models/leaf_disease_model.keras` and metrics to `models/evaluation_metrics.json`.

5. **Real Model Prediction (No Hardcoded / Fake Results)**:
   - Preprocesses uploaded images and runs model forward pass dynamically.
   - Outputs: Predicted Disease Name, Health Status (Healthy vs Infected), Confidence Score %, and Top-3 Probabilities Distribution.

6. **Disease Knowledge Base & Agronomic Advice**:
   - Displays symptoms, organic/chemical treatment countermeasures, and agricultural prevention best practices.
   - Includes agricultural & AI safety disclaimers.

7. **Agriculture-Inspired Web UI**:
   - Built with Streamlit using a custom lush green theme (`#2e7d32`).
   - Features 4 navigation tabs: *Leaf Health Inspection*, *Model Metrics & Performance*, *Disease Knowledge Base Catalog*, and *System Overview*.

---

## 📁 Project Structure

```
leaf-disease-detection/
│
├── dataset/                    # Dataset split directory
│   ├── train/                  # Training images per category class folder
│   ├── validation/             # Validation images
│   └── test/                   # Test images for evaluation
│
├── models/                     # Saved models and evaluation output artifacts
│   ├── leaf_disease_model.keras # Saved Keras deep learning model file
│   ├── class_names.json        # Categorical class index mapping
│   ├── evaluation_metrics.json # Saved accuracy, precision, recall, F1 JSON
│   ├── training_history.png    # Training vs Validation loss & accuracy graph
│   └── confusion_matrix.png   # Confusion Matrix plot
│
├── notebooks/                  # Interactive Jupyter Notebooks
│   └── model_training.ipynb    # Walkthrough notebook for training & evaluation
│
├── src/                        # Modular Python source package
│   ├── __init__.py
│   ├── preprocessing.py        # Image validation, resizing & augmentation
│   ├── prediction.py           # Model loading & inference prediction engine
│   └── disease_info.py         # Disease Knowledge Base & metadata lookup
│
├── app.py                      # Streamlit frontend web application
├── train.py                    # Standalone model training & evaluation script
├── generate_sample_dataset.py  # Synthetic leaf dataset generator for quick testing
├── requirements.txt            # Python library dependencies
└── README.md                   # Complete project documentation & setup guide
```

---

## 🛠️ Technology Stack

- **Programming Language**: Python 3.9+
- **Deep Learning Framework**: TensorFlow / Keras (MobileNetV2 / CNN)
- **Image Processing**: OpenCV, Pillow (PIL), NumPy
- **Data Science & Evaluation**: Scikit-Learn, Pandas, Matplotlib, Seaborn
- **Web Application**: Streamlit

---

## 🚀 Installation & Setup Instructions

### 1. Clone or Open the Repository
Navigate to the project root folder:
```bash
cd "Leaf Disease Detection"
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🌿 Dataset Setup Guide

This project supports standard plant disease datasets such as **PlantVillage**.

### Option A: Out-of-the-Box Sample Dataset Generator (Quickest)
You can generate a sample dataset immediately by running:
```bash
python generate_sample_dataset.py
```
This automatically populates `dataset/train`, `dataset/validation`, and `dataset/test` with sample leaf images across 13 plant/disease categories.

### Option B: Using the Full PlantVillage Dataset
1. Download the **PlantVillage** dataset from Kaggle or GitHub:
   - [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantvillage-dataset)
2. Extract the dataset into `dataset/` directory according to the split structure:
   ```
   dataset/
   ├── train/
   │   ├── Tomato_Early_Blight/
   │   ├── Tomato_Healthy/
   │   └── ...
   ├── validation/
   │   └── ...
   └── test/
       └── ...
   ```

---

## 🏋️ Model Training & Evaluation

To train the deep learning model and calculate performance metrics, run:
```bash
python train.py
```

### What `train.py` Does:
1. Loads images from `dataset/train` and `dataset/validation`.
2. Applies data augmentation to training samples.
3. Trains a **MobileNetV2 Transfer Learning** model for 10 epochs.
4. Evaluates test set performance and prints:
   - Test Accuracy & Loss
   - Classification Report (Precision, Recall, F1-Score)
   - Confusion Matrix
5. Exports artifacts to `models/`:
   - `leaf_disease_model.keras`
   - `class_names.json`
   - `training_history.png`
   - `confusion_matrix.png`
   - `evaluation_metrics.json`

---

## 🖥️ Running the Streamlit Web Application

To launch the web interface:
```bash
streamlit run app.py
```

Once executed, open your browser at `http://localhost:8501`.

### App Features:
- **Leaf Health Inspection**: Upload any leaf photo, click **Analyze Leaf Health**, and view instant health status, disease name, confidence score %, symptoms, treatment options, and top class probabilities.
- **Model Metrics & Performance**: View saved training curves, accuracy/loss graphs, weighted precision/recall/F1-score, and confusion matrix.
- **Disease Knowledge Base**: Searchable catalog of all plant diseases and treatments.

---

## 📊 Example Prediction Output

```text
Prediction: Tomato Early Blight
Status: Infected ⚠️
Confidence: 94.27%
Plant Host: Tomato

Symptoms:
- Concentric dark brown spots ('target-board' pattern) on older leaves.
- Yellow halo surrounding leaf lesions.

Treatment:
- Apply copper-based fungicides or chlorothalonil at early symptom onset.
- Prune infected lower leaves to restrict fungal spore splash.
```

If Healthy:
```text
Prediction: Healthy Leaf
Status: Healthy ✅
Confidence: 97.10%
Plant Host: Tomato

Description: The leaf exhibits vibrant green coloration and no visible signs of infection.
```

---

## ⚠️ Disclaimer

This application uses deep learning computer vision models for decision support and educational demonstration. Diagnostic predictions should be verified with qualified agricultural experts or plant pathology laboratory testing.
