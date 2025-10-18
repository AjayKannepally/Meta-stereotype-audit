# xai_utils.py
import torch
import numpy as np
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import shap
from PIL import Image
from torchvision import transforms
import joblib


# -----------------------------------------------------------
# IMAGE XAI FUNCTIONS
# -----------------------------------------------------------

def compute_gradcam_for_image(model, img_tensor, target_layer=None):
    """Compute GradCAM heatmap for an image tensor."""
    if target_layer is None:
        # Automatically detect a convolution layer
        if hasattr(model, 'features'):
            target_layer = model.features[-1]
        else:
            target_layer = list(model.children())[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0))[0]
    return grayscale_cam  # (H, W)


def gradcam_meta_features(grayscale_cam):
    """Extract simple meta features from GradCAM heatmap."""
    flat = grayscale_cam.flatten()
    flat = flat / (flat.sum() + 1e-9)  # normalize for stability

    features = {
        'cam_entropy': -np.sum((flat + 1e-9) * np.log(flat + 1e-9)),
        'cam_mean': flat.mean(),
        'cam_std': flat.std(),
        'cam_max': flat.max(),
        'cam_sparsity': np.sum(flat > np.percentile(flat, 90)) / flat.size
    }
    return features


# -----------------------------------------------------------
# TEXT XAI FUNCTIONS
# -----------------------------------------------------------

def shap_text_meta(clf, embedder, texts, n_background=50):
    """Compute SHAP meta-features for text embeddings safely."""
    X = embedder.encode(texts)
    background = shap.sample(X, n_background)

    explainer = shap.KernelExplainer(lambda x: clf.predict_proba(x), background)
    shap_vals = explainer.shap_values(X, nsamples=50)

    meta = []
    for i in range(len(texts)):
        abs_vals = []
        for cls_vals in shap_vals:
            safe_i = min(i, cls_vals.shape[0] - 1)  # Avoid index errors
            abs_vals.append(np.abs(cls_vals[safe_i]).mean())
        avg_abs = np.mean(abs_vals)
        meta.append({"text_shap_abs_mean": float(avg_abs)})
    return meta


# -----------------------------------------------------------
# IMAGE META FEATURE EXTRACTION
# -----------------------------------------------------------

def extract_image_meta_features(model, label_map, image_csv, image_dir, device='cpu'):
    """Extract GradCAM-based meta features for a dataset of images."""
    df = pd.read_csv(image_csv)
    trans = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    rows = []
    model.eval()

    for _, r in df.iterrows():
        img_path = f"{image_dir}/{r['filename']}"
        img = Image.open(img_path).convert("RGB")
        t = trans(img)

        with torch.no_grad():
            out = model(t.unsqueeze(0).to(device))

        # Grad-CAM
        cam = compute_gradcam_for_image(model, t.to(device))
        feats = gradcam_meta_features(cam)

        # Model confidence
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]
        feats['pred_conf'] = float(prob.max())
        feats['pred_class'] = int(prob.argmax())
        feats['filename'] = r['filename']
        feats['true_label'] = r['profession']
        rows.append(feats)

    return pd.DataFrame(rows)


