import pandas as pd

df_img = pd.read_csv("data/meta/image_meta.csv")
df_txt = pd.read_csv("data/meta/text_meta.csv")

print(df_img.columns)
print(df_txt.columns)

print(df_img['true_label'].value_counts())
print(df_txt['occupation'].value_counts())



