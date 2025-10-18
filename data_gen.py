# data_gen.py
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import string

# Config
OUT_DIR = "data"
IMG_DIR = os.path.join(OUT_DIR, "images")
TXT_DIR = os.path.join(OUT_DIR, "text")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

# ---------- Image synthetic dataset ----------
# Create synthetic "profession" images: a simple colored rectangle (person silhouette placeholder)
# Add a correlated background color to induce spurious correlation (e.g., blue -> "doctor", red -> "nurse")

PROFESSIONS = ["doctor", "nurse", "engineer"]
BG_COLORS = {"doctor":"#a6cee3", "nurse":"#fb9a99", "engineer":"#b2df8a"}  # correlated backgrounds

def create_image(profession, idx, biased=True):
    # image size
    W, H = 128, 128
    bg_color = BG_COLORS[profession] if biased else random.choice(list(BG_COLORS.values()))
    im = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(im)
    # draw a circle as person silhouette in center with slight attribute color based on profession
    fill_colors = {"doctor":"#fdbf6f", "nurse":"#cab2d6", "engineer":"#ffff99"}
    draw.ellipse((30, 20, 98, 108), fill=fill_colors[profession])
    # draw label small
    draw.text((5,5), profession, fill=(0,0,0))
    fname = f"{profession}_{idx}.png"
    im.save(os.path.join(IMG_DIR, fname))
    return fname

def gen_images(n_per_class=200, biased=True):
    rows = []
    for p in PROFESSIONS:
        for i in range(n_per_class):
            fname = create_image(p, i, biased=biased)
            rows.append({"filename": fname, "profession": p, "biased_bg": biased})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(IMG_DIR, "labels.csv"), index=False)
    print("Image dataset created:", len(df))
    return df

# ---------- Text synthetic dataset (resumes-like) ----------
MALE_NAMES = ["John", "Michael", "David", "James", "Robert"]
FEMALE_NAMES = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth"]
OCCUPATIONS = ["engineer","nurse","manager","developer","scientist"]

def create_resume(name, occupation, idx):
    # create a simple text example; include patterns to induce bias
    template = (
        f"Name: {name} {random.choice(['Smith','Johnson','Lee','Kumar','Garcia'])}\n"
        f"Summary: Experienced {occupation} with expertise in project management and team leadership.\n"
        f"Skills: Python, Data Analysis, Communication, Problem Solving\n"
    )
    fname = f"resume_{idx}.txt"
    with open(os.path.join(TXT_DIR, fname),"w",encoding="utf-8") as f:
        f.write(template)
    return fname, template

def gen_resumes(n=500, bias_ratio=0.7):
    rows=[]
    for i in range(n):
        # deliberately correlate some occupations with gendered names
        if random.random() < bias_ratio:
            # biased mapping: nurse -> female name, engineer -> male name
            occ = random.choice(OCCUPATIONS)
            if occ in ["nurse","manager"]:
                name = random.choice(FEMALE_NAMES)
            else:
                name = random.choice(MALE_NAMES)
        else:
            name = random.choice(MALE_NAMES+FEMALE_NAMES)
            occ = random.choice(OCCUPATIONS)
        fname, txt = create_resume(name, occ, i)
        rows.append({"filename": fname, "text": txt, "name": name, "occupation": occ})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TXT_DIR,"labels.csv"), index=False)
    print("Resume dataset created:", len(df))
    return df

if __name__ == "__main__":
    gen_images(n_per_class=300, biased=True)
    gen_resumes(n=600, bias_ratio=0.8)
