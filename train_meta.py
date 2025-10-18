import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load meta features
df_img = pd.read_csv("data/meta/image_meta.csv")
df_txt = pd.read_csv("data/meta/text_meta.csv")

# Features and labels for image
X_img = df_img.drop(columns=['filename','true_label']).values
y_img = df_img['true_label'].astype('category').cat.codes.values

# Features and labels for text
X_txt = df_txt.drop(columns=['occupation']).values
y_txt = df_txt['occupation'].astype('category').cat.codes.values

# Train classifiers
clf_img = GradientBoostingClassifier()
clf_img.fit(X_img, y_img)

clf_txt = GradientBoostingClassifier()
clf_txt.fit(X_txt, y_txt)

# Save models
joblib.dump(clf_img, "models/meta_image_clf.joblib")
joblib.dump(clf_txt, "models/meta_text_clf.joblib")

print("✅ Trained and saved meta classifiers")
