# meta_feature_extractor.py
import os
import joblib
import torch
import pandas as pd
from xai_utils import extract_image_meta_features, shap_text_meta

# Paths for models and data
IMAGE_MODEL = "models/base_image_model.pth"
IMAGE_CSV = "data/images/labels.csv"
IMAGE_DIR = "data/images"

TEXT_MODEL = "models/base_text_model.joblib"
TEXT_CSV = "data/text/labels.csv"

# Output directory
META_DIR = "data/meta"
os.makedirs(META_DIR, exist_ok=True)


def run_image_meta(device='cpu'):
    import train_base_image as tbi  # import CNN definition
    print("Extracting image meta features...")

    # Load trained CNN model
    state = torch.load(IMAGE_MODEL, map_location=device)
    model = tbi.SmallCNN(n_classes=3)
    model.load_state_dict(state['model_state'])
    model.to(device)

    # Extract image meta features
    df_meta = extract_image_meta_features(model, state.get('label_map', {}), IMAGE_CSV, IMAGE_DIR, device=device)
    df_meta.to_csv(os.path.join(META_DIR, "image_meta.csv"), index=False)
    print(f"✅ Saved image meta features to {os.path.join(META_DIR, 'image_meta.csv')}")


def run_text_meta():
    obj = joblib.load(TEXT_MODEL)
    clf = obj['clf']
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(obj['embedder_model'])
    df = pd.read_csv(TEXT_CSV)
    texts = df['text'].tolist()
    meta = shap_text_meta(clf, embedder, texts, n_background=50)
    
    # Include the original labels in the meta CSV
    df_meta_with_labels = pd.concat([df[['occupation']], pd.DataFrame(meta)], axis=1)
    os.makedirs(META_DIR, exist_ok=True)
    df_meta_with_labels.to_csv(os.path.join(META_DIR, "text_meta.csv"), index=False)
    print(f"✅ Saved text meta features with labels to {os.path.join(META_DIR, 'text_meta.csv')}")


if __name__ == "__main__":
    run_image_meta(device='cpu')
    run_text_meta()




