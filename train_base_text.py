# train_base_text.py
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib

# Directories and paths
TXT_DIR = "data/text"
CSV = os.path.join(TXT_DIR, "labels.csv")
MODEL_OUT = "models/base_text_model.joblib"
os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv(CSV)
texts = df['text'].tolist()

# Encode labels as integers
labels = df['occupation'].astype('category')
label_map = dict(enumerate(labels.cat.categories))        # int -> label
inv_label_map = {v: k for k, v in label_map.items()}      # label -> int
y = labels.cat.codes.values                               # numeric labels

# Generate embeddings using a small SBERT model
embedder = SentenceTransformer('all-MiniLM-L6-v2')        # small, fast
X = embedder.encode(texts, show_progress_bar=True)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# Save model + embedder info
joblib.dump({
    "clf": clf,
    "embedder_model": "all-MiniLM-L6-v2",
    "label_map": label_map
}, MODEL_OUT)

print("Saved text model:", MODEL_OUT)

