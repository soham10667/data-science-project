"""
Leaf Disease Detection – Image-Based Plant Health Inspection System
Streamlit User Interface Application
"""

import os
import json
import time
import pandas as pd
import streamlit as st
from PIL import Image

# Import custom src modules
from src.preprocessing import validate_image
from src.prediction import LeafDiseasePredictor, get_predictor
from src.disease_info import DISEASE_KNOWLEDGE_BASE, DISCLAIMER_TEXT

# Configure Streamlit Page
st.set_page_config(
    page_title="Leaf Health Inspection System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling - Theme Adaptive Agriculture Green Theme
CUSTOM_CSS = """
<style>
    /* Header Banner Styling */
    .header-container {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
        padding: 2rem;
        border-radius: 14px;
        color: #ffffff !important;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        color: #ffffff !important;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.05rem;
        color: #e8f5e9 !important;
        margin-top: 0.4rem;
        opacity: 0.95;
    }
    
    /* Card Container - Theme Adaptive */
    .card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(46, 125, 50, 0.3);
        margin-bottom: 1.5rem;
    }
    
    /* Result Badges */
    .status-badge-healthy {
        background-color: #1b5e20;
        color: #ffffff !important;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    
    .status-badge-infected {
        background-color: #c62828;
        color: #ffffff !important;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }

    .confidence-metric {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4caf50;
    }
    
    /* Disclaimer Card */
    .disclaimer-box {
        background-color: rgba(251, 192, 45, 0.15);
        border-left: 4px solid #fbc02d;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        font-size: 0.92rem;
        margin-top: 1.5rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Helper function to load model metrics
def load_saved_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), "models", "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/leaf.png", width=70)
st.sidebar.title("Plant Health AI")
st.sidebar.caption("Image-Based Disease Diagnostics")

page_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "🔍 Leaf Health Inspection",
        "📊 Model Metrics & Performance",
        "📚 Disease Knowledge Base",
        "ℹ️ System Overview"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### 🌾 Quick Status")
predictor = get_predictor()
if predictor.is_model_loaded():
    st.sidebar.success(f"Model Status: Active & Loaded ({predictor.backend.upper()})")
else:
    st.sidebar.warning("Model Status: Not Loaded (Run train.py)")

# --- PAGE 1: LEAF HEALTH INSPECTION ---
if page_selection == "🔍 Leaf Health Inspection":
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">🌱 Leaf Disease Detection & Inspection System</h1>
            <p class="header-subtitle">Upload a plant leaf image to analyze health status, identify specific diseases, and view treatment recommendations.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("1. Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Select a leaf photo (JPG, JPEG, PNG format)",
            type=["jpg", "jpeg", "png"],
            help="For best accuracy, ensure the leaf is well-lit and centered."
        )

        if uploaded_file is not None:
            # Validate image format and corruption
            is_valid, val_result = validate_image(uploaded_file)
            
            if not is_valid:
                st.error(f"❌ Invalid Image: {val_result}")
            else:
                image = val_result
                st.image(image, caption="Uploaded Leaf Preview", use_container_width=True)
                
                # Image Metadata Info
                w, h = image.size
                st.caption(f"📏 Dimensions: {w} × {h} pixels | Format: RGB | Status: Validated")
                
                analyze_btn = st.button("🔬 Analyze Leaf Health", type="primary", use_container_width=True)
        else:
            st.info("💡 Please upload an image to begin inspection.")
            # Sample demo image recommendation
            sample_dir = os.path.join(os.path.dirname(__file__), "dataset", "test")
            if os.path.exists(sample_dir):
                st.markdown("**Or test with a sample image from dataset:**")
                categories = [d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))]
                if categories:
                    selected_cat = st.selectbox("Select Sample Category", categories)
                    cat_path = os.path.join(sample_dir, selected_cat)
                    sample_files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if sample_files:
                        sample_img_path = os.path.join(cat_path, sample_files[0])
                        if st.button("Load Sample Image"):
                            with open(sample_img_path, "rb") as f:
                                st.session_state["sample_image_path"] = sample_img_path
                                st.rerun()

    with col2:
        st.subheader("2. Inspection Results & Diagnosis")
        
        # Process analysis trigger
        trigger_prediction = False
        target_image = None
        
        if uploaded_file is not None and 'analyze_btn' in locals() and analyze_btn:
            trigger_prediction = True
            target_image = uploaded_file
        elif "sample_image_path" in st.session_state:
            trigger_prediction = True
            target_image = st.session_state["sample_image_path"]
            del st.session_state["sample_image_path"]

        if trigger_prediction and target_image is not None:
            with st.spinner("Analyzing leaf patterns using Deep Learning model..."):
                time.sleep(0.6) # Brief smooth UI transition delay
                result = predictor.predict(target_image)

            if not result.get("success", False):
                st.error(f"⚠️ {result.get('error', 'Prediction failed.')}")
            else:
                # Prediction Data
                pred_class = result["predicted_class"]
                status = result["status"]
                confidence = result["confidence"]
                info = result["disease_info"]

                # Result Header Card
                badge_class = "status-badge-healthy" if status == "Healthy" else "status-badge-infected"
                badge_icon = "✅" if status == "Healthy" else "⚠️"
                
                st.markdown(f"""
                    <div class="card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="{badge_class}">{badge_icon} Plant Status: {status}</span>
                                <h2 style="margin: 0.8rem 0 0.2rem 0; color: #1b5e20;">Detected: {info['disease']}</h2>
                                <p style="margin:0; color: #64748b; font-weight: 600;">Plant Species: {info['plant']}</p>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 0.85rem; color: #64748b;">Confidence Score</div>
                                <div class="confidence-metric">{confidence:.2f}%</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                if status == "Healthy":
                    st.success("🌱 **Plant Status: Healthy Leaf**")
                    st.markdown(f"**Confidence:** `{confidence:.2f}%`")
                    st.markdown("✅ **Condition:** No disease symptoms detected. Leaf is in optimal health.")
                else:
                    # Clean Section Breakdown for Infected Leaves
                    st.markdown(f"### 🦠 {info['disease']}")
                    st.markdown(f"**Possible Cause / Description:** {info['description']}")
                    st.markdown(f"**Severity Level:** `{info['severity']}`")

                    st.divider()
                    col_sym, col_trt = st.columns(2)
                    with col_sym:
                        st.markdown("#### 🔍 Symptoms")
                        for sym in info.get("symptoms", []):
                            st.markdown(f"- {sym}")

                    with col_trt:
                        st.markdown("#### 💊 Treatment & Action Plan")
                        for trt in info.get("treatment", []):
                            st.markdown(f"- 🌿 {trt}")

                    st.divider()
                    st.markdown("#### 🛡️ Agricultural Prevention Best Practices")
                    for prv in info.get("prevention", []):
                        st.markdown(f"- {prv}")

                st.divider()
                st.markdown("#### 📊 Confidence Breakdown across Top Classes")
                top3 = result.get("top_3_predictions", [])
                for item in top3:
                    st.write(f"**{item['class']}** ({item['confidence']}%)")
                    st.progress(min(int(item['confidence']), 100))

                # Disclaimer Note
                st.markdown(f"""
                    <div class="disclaimer-box">
                        {DISCLAIMER_TEXT}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👈 Upload an image on the left and click **Analyze Leaf Health** to view diagnostic results.")

# --- PAGE 2: MODEL METRICS & PERFORMANCE ---
elif page_selection == "📊 Model Metrics & Performance":
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">📊 Deep Learning Model Performance & Evaluation</h1>
            <p class="header-subtitle">Training history, loss/accuracy curves, classification metrics, and confusion matrix visualizer.</p>
        </div>
    """, unsafe_allow_html=True)

    metrics = load_saved_metrics()
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    history_plot = os.path.join(models_dir, "training_history.png")
    cm_plot = os.path.join(models_dir, "confusion_matrix.png")

    if metrics is not None:
        # Top KPI Metrics Cards
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Overall Test Accuracy", f"{metrics['test_accuracy']*100:.2f}%")
        mcol2.metric("Weighted Precision", f"{metrics['precision']*100:.2f}%")
        mcol3.metric("Weighted Recall", f"{metrics['recall']*100:.2f}%")
        mcol4.metric("Weighted F1-Score", f"{metrics['f1_score']*100:.2f}%")

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📈 Training vs Validation Accuracy & Loss")
            if os.path.exists(history_plot):
                st.image(history_plot, use_container_width=True)
            else:
                st.info("Accuracy plot figure not available.")

        with col_right:
            st.subheader("🧩 Confusion Matrix")
            if os.path.exists(cm_plot):
                st.image(cm_plot, use_container_width=True)
            else:
                st.info("Confusion matrix figure not available.")

        st.divider()
        st.subheader("📋 Classification Report per Category")
        if "classification_report" in metrics:
            report_dict = metrics["classification_report"]
            rows = []
            for cls_key, scores in report_dict.items():
                if isinstance(scores, dict):
                    rows.append({
                        "Category Class": cls_key.replace("_", " "),
                        "Precision": f"{scores.get('precision', 0)*100:.1f}%",
                        "Recall": f"{scores.get('recall', 0)*100:.1f}%",
                        "F1-Score": f"{scores.get('f1-score', 0)*100:.1f}%",
                        "Support": int(scores.get('support', 0))
                    })
            report_df = pd.DataFrame(rows)
            st.dataframe(report_df, use_container_width=True, hide_index=True)

    else:
        st.warning("⚠️ No saved evaluation metrics found. Please train the model first by running `python train.py`.")
        st.markdown("""
            **To train the model:**
            1. Open your terminal in the project directory.
            2. Run: `python train.py`
            3. Refresh this page after training completes.
        """)

# --- PAGE 3: DISEASE KNOWLEDGE BASE ---
elif page_selection == "📚 Disease Knowledge Base":
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">📚 Plant Disease Knowledge Base Catalog</h1>
            <p class="header-subtitle">Browse symptoms, causes, organic treatments, and prevention guidelines across all supported plant species.</p>
        </div>
    """, unsafe_allow_html=True)

    search_query = st.text_input("🔍 Search disease by name or plant species", "")

    for key, info in DISEASE_KNOWLEDGE_BASE.items():
        if search_query.lower() in info['disease'].lower() or search_query.lower() in info['plant'].lower():
            status_color = "🟢" if info['status'] == "Healthy" else "🔴"
            with st.expander(f"{status_color} {info['plant']} - {info['disease']} ({info['status']})"):
                st.write(f"**Severity:** {info['severity']}")
                st.write(f"**Description:** {info['description']}")
                
                st.markdown("**Key Symptoms:**")
                for s in info.get('symptoms', []):
                    st.write(f"- {s}")
                    
                st.markdown("**Treatment Options:**")
                for t in info.get('treatment', []):
                    st.write(f"- 🌿 {t}")

# --- PAGE 4: SYSTEM OVERVIEW ---
elif page_selection == "ℹ️ System Overview":
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">ℹ️ System Architecture & Technology Stack</h1>
            <p class="header-subtitle">Overview of the machine learning pipeline, data flow, and technologies powering this application.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🔄 Processing Workflow Architecture
    
    ```
    Plant Leaf Image Upload (JPG / JPEG / PNG)
                      ↓
           Image Preprocessing & RGB Validation (224 × 224)
                      ↓
       Data Normalization & Transfer Learning Model (MobileNetV2)
                      ↓
       Softmax Categorical Classification (Healthy vs Infected)
                      ↓
          Confidence Computation & Disease KB Mapping
                      ↓
     Interactive Streamlit UI Result Card & Agronomic Recommendations
    ```
    
    ### 💻 Technology Stack
    - **Language:** Python 3.9+
    - **Deep Learning Framework:** TensorFlow / Keras (MobileNetV2 Transfer Learning)
    - **Image Processing:** OpenCV, Pillow (PIL), NumPy
    - **Data Analysis & Evaluation:** Scikit-learn, Pandas, Matplotlib, Seaborn
    - **Frontend User Interface:** Streamlit
    """)
