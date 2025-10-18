# app.py
import streamlit as st
import joblib
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from xai_utils import compute_gradcam_for_image, gradcam_meta_features

st.set_page_config(layout="centered", page_title="Meta-AI Bias Auditor")
st.title("Meta-AI Stereotype Reliance Auditor (Image & Text)")

# Paths
IMG_META_MODEL_PATH = "models/meta_image_clf.joblib"
BASE_IMG_MODEL = "models/base_image_model.pth"
TXT_META_MODEL_PATH = "models/meta_text_clf.joblib"
TEXT_BASE_MODEL = "models/base_text_model.joblib"

@st.cache_resource
def load_models():
    img_meta = joblib.load(IMG_META_MODEL_PATH)
    img_base = torch.load(BASE_IMG_MODEL, map_location='cpu')
    txt_meta = joblib.load(TXT_META_MODEL_PATH)
    txt_base = joblib.load(TEXT_BASE_MODEL)
    return img_meta, img_base, txt_meta, txt_base

img_meta, img_base, txt_meta, txt_base = load_models()

mode = st.sidebar.radio("Mode", ["Image Audit", "Text Audit"])

# ---------------- IMAGE AUDIT ----------------
if mode == "Image Audit":
    uploaded = st.file_uploader("Upload an image of person/profession", type=['png', 'jpg', 'jpeg'])
    if uploaded:
        img = Image.open(uploaded).convert("RGB").resize((64, 64))
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Prepare tensor
        transform = transforms.Compose([transforms.ToTensor()])
        t = transform(img)

        # Load base CNN and its label map
        import train_base_image as tbi
        checkpoint = torch.load(BASE_IMG_MODEL, map_location='cpu')
        model = tbi.SmallCNN(n_classes=len(checkpoint['label_map']))
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        # Get label map from model
        inv_label_map = {v: k for k, v in checkpoint['label_map'].items()}

        # Predict
        with torch.no_grad():
            out = model(t.unsqueeze(0))
        prob = torch.softmax(out, dim=1).numpy()[0]
        pred_idx = int(prob.argmax())
        pred_label = inv_label_map[pred_idx]
        confidence = prob[pred_idx] * 100

        # Display prediction
        st.markdown(f"### 🧠 Prediction: **{pred_label.capitalize()}**")
        st.markdown(f"### 📊 Confidence: **{confidence:.1f}%**")

        # Grad-CAM visualization
        cam = compute_gradcam_for_image(model, t)
        feats = gradcam_meta_features(cam)
        st.write("Meta features extracted:", feats)

        # Meta model prediction
        feat_vec = np.array(list(feats.values())).reshape(1, -1)
        risk = img_meta['clf'].predict_proba(feat_vec)[0][1]
        st.metric("Stereotype Reliance Risk", f"{risk*100:.1f}%")

        # Visualize heatmap
        plt.imshow(cam, cmap='jet')
        plt.title("Grad-CAM (Heatmap)")
        plt.axis('off')
        st.pyplot(plt)

# ---------------- TEXT AUDIT ----------------
elif mode == "Text Audit":
    txt = st.text_area("Paste resume / text here", height=200)

    if st.button("Analyze Text"):
        if txt.strip() == "":
            st.warning("Please paste some text to analyze.")
        else:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(txt_base['embedder_model'])
            X = embedder.encode([txt])
            clf = txt_meta['clf']

            shap_mean = float(abs(X).mean())
            feat_vec = [[shap_mean]]
            risk = txt_meta['clf'].predict_proba(feat_vec)[0][1]

            st.metric("Stereotype Reliance Risk (Text)", f"{risk*100:.1f}%")
            st.write("Proxy SHAP Mean:", shap_mean)
            st.info("💡 Recommendation: Remove sensitive identifiers like names, gender words, or demographics before using in AI systems.")
