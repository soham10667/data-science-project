# 🌿 Leaf Disease Detection

An end-to-end Computer Vision & Deep Learning web application built with **TensorFlow / Keras** and **Streamlit** to detect plant leaf diseases and provide treatment guidelines.

---

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Sample Dataset & Train Model (Optional)**
   ```bash
   python generate_sample_dataset.py
   python train.py
   ```

3. **Run Streamlit Web App**
   ```bash
   streamlit run app.py
   ```

---

## ✨ Key Features

- **Leaf Image Inspection**: Upload leaf images (`JPG`, `PNG`) for real-time classification.
- **Deep Learning Model**: Uses MobileNetV2 Transfer Learning for leaf health diagnosis (Healthy vs Infected).
- **Treatment Guidelines**: Displays symptoms, organic countermeasures, and prevention tips for detected diseases.
- **Performance Analytics**: View confusion matrix, accuracy/loss curves, and evaluation metrics.

---

## 📁 Project Structure

```
Leaf Disease Detection/
├── app.py                      # Streamlit Web Application
├── train.py                    # Model Training Pipeline (MobileNetV2 / CNN)
├── generate_sample_dataset.py  # Utility script to generate sample dataset
├── requirements.txt            # Python dependencies
├── dataset/                    # Plant leaf image dataset
├── models/                     # Saved model (.keras) & metrics (.json)
├── notebooks/                  # Jupyter notebook for experimentation
└── src/                        # Data preprocessing & evaluation modules
```

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Deep Learning**: TensorFlow / Keras, MobileNetV2
- **Data Processing & Viz**: NumPy, Pillow, Scikit-Learn, Matplotlib, Seaborn

