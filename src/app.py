from io import BytesIO
from pathlib import Path
import warnings
import torch
import streamlit as st
import re
from PIL import Image
from utils import (
    MODEL_DIR, REPORT_DIR, get_model, get_transforms, load_class_names,
    GradCAM, generate_gradcam_overlay,
)

# Set page config
st.set_page_config(
    page_title="AI Document Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .header-banner {
        padding: 30px;
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(124, 58, 237, 0.08) 50%, rgba(236, 72, 153, 0.08) 100%);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.02);
    }
    
    .title-text {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4f46e5, #9333ea, #ec4899);
        -webkit-background-clip: text;
        margin: 0;
        padding-bottom: 6px;
        letter-spacing: -0.5px;
    }
    
    .subtitle-text {
        font-size: 1.15rem;
        color: #718096;
        margin-top: 6px;
        margin-bottom: 0px;
        font-weight: 400;
    }
    
    .interactive-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.02);
        backdrop-filter: blur(8px);
        margin-bottom: 25px;
    }
    
    .sidebar-stats-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 16px;
        margin-bottom: 16px;
    }
    
    .sidebar-stat-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #a0aec0;
        margin-bottom: 4px;
        font-weight: 600;
    }
    
    .sidebar-stat-val {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .pred-box {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .pred-class {
        font-size: 2.6rem;
        font-weight: 700;
        margin: 6px 0;
        letter-spacing: -0.5px;
    }
    
    .pred-confidence {
        font-size: 1.25rem;
        opacity: 0.95;
        font-weight: 500;
    }
    
    .progress-container {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        transition: background-color 0.2s ease;
    }
    .progress-container:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    .progress-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.95rem;
        margin-bottom: 5px;
        font-weight: 600;
    }
    .progress-bar-bg {
        background-color: rgba(226, 232, 240, 0.15);
        border-radius: 10px;
        height: 10px;
        width: 100%;
        overflow: hidden;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease-in-out;
    }
            
    [data-testid="stFileUploader"] {
        border-radius: 18px;
        padding: 20px;
        background: rgba(99,102,241,0.04);
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1;
        background: rgba(99,102,241,0.08);
    }

    [data-testid="stFileUploader"] section {
        padding: 20px;
    }

    [data-testid="stFileUploader"] small {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Keep uploads modest for a local demonstration and decode only the first TIFF frame.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def load_uploaded_image(uploaded_file):
    if Path(uploaded_file.name).suffix.lower() not in {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}:
        raise ValueError('Choose a PNG, JPG, JPEG, TIF, or TIFF image.')
    data = uploaded_file.getvalue()
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError('Choose an image smaller than 20 MB.')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError('Choose an image with at most 20 million pixels.')
            image = source.convert('RGB')
            image.load()
    return image


@st.cache_resource
def load_classifier():
    class_names = load_class_names()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=len(class_names), pretrained=False)
    checkpoint = torch.load(MODEL_DIR / 'document_classifier.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    model.eval()
    return model, class_names, device


# Render top header banner
st.markdown("""
<div class='header-banner'>
    <h1 class='title-text'>Automated Document Classifier</h1>
    <p class='subtitle-text'>Document image classification with MobileNetV2 and Grad-CAM visualization.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Reported Test Accuracy", "85.1%")

with col2:
    st.metric("Classes", "10")

with col3:
    st.metric("Dataset Images", "3482")

with col4:
    st.metric("Architecture", "MobileNetV2")

missing = [p.name for p in (MODEL_DIR / 'document_classifier.pth', MODEL_DIR / 'class_mapping.json') if not p.is_file()]
if missing:
    st.error('Missing classifier files in models/: ' + ', '.join(missing))

else:
    # Load resources
    try:
        model, class_names, device = load_classifier()
    except Exception:
        st.error('Could not load the classifier. Check that the checkpoint and class mapping are valid and compatible.')
        st.stop()
    _, val_transform = get_transforms()
    
    # Sidebar Setup
    st.sidebar.markdown("### ⚙️System Configuration")
    
    st.sidebar.markdown("""
    <div class='sidebar-stats-card'>
        <div class='sidebar-stat-label'>
            MODEL PERFORMANCE
        </div>
        <div class='sidebar-stat-val'>
            Reported test accuracy: 85.1%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div class='sidebar-stats-card'>
        <div class='sidebar-stat-label'>
            DATASET SIZE
        </div>
        <div class='sidebar-stat-val'>
            3482 Images
        </div>
    </div>
    """, unsafe_allow_html=True)
        
    st.sidebar.markdown(f"""
    <div class='sidebar-stats-card'>
        <div class='sidebar-stat-label'>Network Architecture</div>
        <div class='sidebar-stat-val'>Convolutional Neural Network</div>
        <div class='sidebar-stat-val'># MobileNetV2</div>
    </div>
    <div class='sidebar-stats-card'>
        <div class='sidebar-stat-label'>Outputs / Categories</div>
        <div class='sidebar-stat-val'>{len(class_names)} Classes</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    st.sidebar.markdown("""
    ### Model Details:

    - Framework: PyTorch
    - Explainability: Grad-CAM
    - Transfer Learning
    - CUDA when available, otherwise CPU
    - Image Size: 384×384
    """)
    
    st.sidebar.caption(f"Current device: {device}")
    st.sidebar.caption('Metrics are from supplied evaluation artifacts, not a new evaluation.')
    # Tabs
    tab1, tab2, tab3 = st.tabs([" ♻️Document Analysis", "📊 Model Performance", "ℹ️ About"])
    
    with tab1:
        st.markdown("")
        image = None
        st.markdown("### 📤 Upload Document")

        uploaded_file = st.file_uploader(
            "Upload Document Image",
            type=["png", "jpg", "jpeg", "tiff", "tif"],
            label_visibility="collapsed"
        )
        st.caption('PNG/JPEG/TIFF, up to 20 MB and 20 million pixels. TIFF: first page only.')
        if uploaded_file is not None:
            try:
                image = load_uploaded_image(uploaded_file)
            except ValueError as e:
                st.error(str(e))
            except Exception:
                st.error('This image could not be decoded. Upload a valid, uncorrupted PNG, JPEG, or TIFF image.')

                
        if image is not None:
            col1, col2 = st.columns([1.1, 0.9])
            
            with col1:
                st.markdown("")
                st.markdown("### Document Analysis & Highlights")
                
                img_input = image
                try:
                    tensor_img = val_transform(img_input).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(tensor_img)
                        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                    top_idx = int(probs.argmax())
                    top_class = class_names[top_idx]
                    top_conf = float(probs[top_idx]) * 100
                except Exception:
                    st.error('Prediction failed. Try another image or restart the application in the documented environment.')
                    st.stop()

                original_resized = img_input.resize((450, 450), Image.Resampling.BILINEAR)
                overlaid_image = None
                cam = None
                with st.spinner('Generating Grad-CAM visualization...'):
                    try:
                        with torch.enable_grad():
                            cam = GradCAM(model, model.features[-1])
                            heatmap, _ = cam(tensor_img.requires_grad_(True), target_category=top_idx)
                        if heatmap.max() > 0:
                            overlaid_image, _ = generate_gradcam_overlay(original_resized, heatmap, alpha=0.45)
                        else:
                            st.info('No positive Grad-CAM activation was found for this prediction.')
                    except Exception:
                        st.warning('The prediction succeeded, but its Grad-CAM visualization could not be generated.')
                    finally:
                        if cam is not None:
                            cam.remove_hooks()

                # Visual view selection
                view_mode = st.segmented_control(
                    "Select Visualization View",
                    options=["Original", "Grad-CAM", "Compare"],
                    default="Compare"
                )
                
                if view_mode is None:
                    view_mode = "Grad-CAM"
                    
                st.markdown("<div style='margin-top:15px; text-align:center;'>", unsafe_allow_html=True)
                if view_mode == "Original":
                    st.image(original_resized, caption="Original Document Layout", use_container_width=True)
                elif view_mode == "Grad-CAM" and overlaid_image is not None:
                    st.image(overlaid_image, caption="Neural Activation Heatmap (Grad-CAM)", use_container_width=True)
                else:
                    col_side1, col_side2 = st.columns(2)
                    with col_side1:
                        st.image(original_resized, caption="Original", use_container_width=True)
                    with col_side2:
                        if overlaid_image is not None:
                            st.image(overlaid_image, caption="Grad-CAM", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                if overlaid_image is not None:
                    st.markdown("""
                    <div style='font-size: 0.9rem; color: #718096; margin-top: 15px; border-left: 3px solid #6366f1; padding-left: 10px;'>
                        <strong>Grad-CAM</strong>: Warm colors indicate regions with stronger positive contributions to the selected class score. This is an approximate visual explanation, not evidence of semantic understanding.
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("")
                st.markdown("### Class Prediction:")
                
                # Render prediction box
                st.markdown(f"""
                <div class='pred-box'>

                <div style='font-size:0.85rem;letter-spacing:1.5px;text-transform:uppercase;'>
                Prediction
                </div>

                <div class='pred-class'>
                {top_class}
                </div>

                <div class='pred-confidence'>
                Class score: {top_conf:.2f}%
                </div>

                <hr style="margin:15px 0;opacity:.3;">

                <div style="
                display:flex;
                justify-content:space-around;
                font-size:.9rem;
                ">

                <div>
                <b>Model</b><br>
                MobileNetV2
                </div>

                <div>
                <b>Classes</b><br>
                10
                </div>

                </div>

                </div>
                """, unsafe_allow_html=True)
                
                st.caption("Softmax scores are not calibrated confidence. Every input is assigned to one of the ten known classes.")
                # Show the two highest scores.
                st.markdown("#### Top Class Scores:")
                sorted_indices = probs.argsort()[::-1][:2]

                for idx in sorted_indices:
                    c_name = class_names[idx]
                    c_prob = probs[idx]
                    
                    st.markdown(f"""
                    <div class='progress-container'>
                        <div class='progress-label'>
                            <span>{c_name}</span>
                            <span>{c_prob*100:.2f}%</span>
                        </div>
                        <div class='progress-bar-bg'>
                            <div class='progress-bar-fill' style='width: {c_prob*100}%;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
            text-align:center;
            padding:60px;
            opacity:0.7;
            ">
                <h3>Ready for Analysis</h3>
                <p>
                Upload a document image to get started.
                </p>
            </div>
            """, unsafe_allow_html=True)   

    with tab2:
        st.markdown('### Model Evaluation Summary')
        st.caption('Saved evaluation artifacts supplied with the project. These results were not newly reproduced.')
        col_a, col_b = st.columns(2)
        for column, name, caption in (
            (col_a, 'confusion_matrix.png', 'Confusion Matrix'),
            (col_b, 'precision_recall.png', 'Precision-Recall Curves'),
        ):
            with column:
                asset = REPORT_DIR / name
                if asset.is_file():
                    try:
                        st.image(str(asset), caption=caption, use_container_width=True)
                    except Exception:
                        st.warning(f'Could not display {name}.')
                else:
                    st.info(f'Saved {name} is unavailable.')
        history_plot = REPORT_DIR / 'training_curves.png'
        if history_plot.is_file():
            try:
                st.image(str(history_plot), caption='Training & Validation History', use_container_width=True)
            except Exception:
                st.warning('Could not display the saved training curves.')
        else:
            st.info('Saved training curves are unavailable.')
        report_path = REPORT_DIR / 'evaluation_report.md'
        if report_path.is_file():
            try:
                report_content = report_path.read_text(encoding='utf-8')
                clean_report = re.sub(r'!\[.*?\]\(.*?\)', '', report_content)
                st.markdown(clean_report)
            except Exception:
                st.warning('Could not read the saved evaluation report.')
        else:
            st.info('The saved evaluation report is unavailable.')

    with tab3:
        st.markdown('### Project and Dataset Details')
        st.markdown("""
        Image-based document classification without a separate OCR stage.
        The CNN can learn visual features such as layout, typography, shapes, and logos.

        **Tobacco3482** is a benchmark of **3,482 scanned document images** in ten categories,
        drawn from tobacco-industry document collections. The dataset is downloaded separately.

        **Classes:** ADVE (Advertisement), Email, Form, Letter, Memo, News, Note,
        Report, Resume, and Scientific.

        **Implementation:**
        - MobileNetV2 with ImageNet initialization during training.
        - Global average pooling, Dropout(0.2), and Linear(1280, 10).
        - Weighted CrossEntropyLoss and Adam.
        - RGB conversion, resize to **384×384**, tensor conversion, and ImageNet normalization.
        - Approximately 80% / 10% / 10% stratified training / validation / test split.
        - CUDA is selected when available; otherwise inference uses CPU.

        **Limitations:** Inputs always receive one of the ten classes. Softmax scores are not
        calibrated confidence. New document styles may reduce accuracy, and Grad-CAM is an
        approximate visual explanation. The supplied results cannot be independently reproduced
        without the original dataset and matching split manifest.
        """)
