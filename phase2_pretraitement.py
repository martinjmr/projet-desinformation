"""
Phase 2 — Pré-traitement du dataset
====================================
Input  : posts_raw.csv
Output : dataset_clean.csv / train.csv / val.csv / test.csv

Etapes :
1. Nettoyage des textes
2. Suppression des doublons et posts trop courts
3. Rééquilibrage des classes
4. Split train / val / test (70 / 15 / 15)
5. Visualisations
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import os
from collections import Counter

# ------------------------------------------------------------------ #
# pip install pandas scikit-learn matplotlib                          #
# ------------------------------------------------------------------ #
from sklearn.model_selection import train_test_split

print("=" * 55)
print("  PHASE 2 — Pré-traitement du dataset")
print("=" * 55)

# ------------------------------------------------------------------ #
# 1. Chargement
# ------------------------------------------------------------------ #
if not os.path.exists("posts_raw.csv"):
    print("ERREUR : posts_raw.csv introuvable dans ce dossier.")
    exit()

df = pd.read_csv("posts_raw.csv")
print(f"\n[1] Chargement : {len(df)} posts")
print(f"    Colonnes : {list(df.columns)}")
print(f"\n    Distribution initiale :")
print(df['label'].value_counts().to_string())

# ------------------------------------------------------------------ #
# 2. Suppression des doublons et posts trop courts
# ------------------------------------------------------------------ #
avant = len(df)
df = df.drop_duplicates(subset=["post_id"])
df = df.drop_duplicates(subset=["texte"])
df['nb_mots_brut'] = df['texte'].astype(str).apply(lambda x: len(x.split()))
df = df[df['nb_mots_brut'] >= 5]
df = df[df['subreddit'] != 'ActualiteFrance']  # quasi vide
print(f"\n[2] Suppression doublons / posts trop courts / subreddit vide :")
print(f"    {avant} → {len(df)} posts ({avant - len(df)} supprimés)")

# ------------------------------------------------------------------ #
# 3. Nettoyage des textes
# ------------------------------------------------------------------ #
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)              # URLs
    text = re.sub(r'u/\w+|r/\w+', '', text)                   # mentions Reddit
    text = re.sub(r'\[removed\]|\[deleted\]', '', text)        # posts supprimés
    text = re.sub(
        r'[^\w\s\'\-àâçéèêëîïôùûüÿœæÀÂÇÉÈÊËÎÏÔÙÛÜŸŒÆ.,!?;:]',
        ' ', text
    )
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['texte_clean'] = df['texte'].apply(clean_text)

# Recalculer longueur après nettoyage
df['nb_mots'] = df['texte_clean'].apply(lambda x: len(x.split()))
avant = len(df)
df = df[df['nb_mots'] >= 5]
print(f"\n[3] Nettoyage des textes :")
print(f"    {avant} → {len(df)} posts")
print(f"    Longueur moyenne : {df['nb_mots'].mean():.0f} mots")
print(f"    Longueur médiane : {df['nb_mots'].median():.0f} mots")

# ------------------------------------------------------------------ #
# 4. Rééquilibrage des classes (undersampling)
# ------------------------------------------------------------------ #
print(f"\n[4] Distribution avant rééquilibrage :")
print(df['label'].value_counts().to_string())

target = df['label'].value_counts().min()
print(f"\n    Cible par classe : {target} posts")

df_balanced = pd.concat([
    df[df['label'] == label].sample(n=target, random_state=42)
    for label in df['label'].unique()
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n    Distribution après rééquilibrage :")
print(df_balanced['label'].value_counts().to_string())
print(f"    Total : {len(df_balanced)} posts")

# ------------------------------------------------------------------ #
# 5. Split train / val / test  (70 / 15 / 15)
# ------------------------------------------------------------------ #
train, temp = train_test_split(
    df_balanced, test_size=0.30,
    random_state=42, stratify=df_balanced['label']
)
val, test = train_test_split(
    temp, test_size=0.50,
    random_state=42, stratify=temp['label']
)

print(f"\n[5] Splits :")
print(f"    Train : {len(train)} posts")
print(f"    Val   : {len(val)}   posts")
print(f"    Test  : {len(test)}  posts")

# Vérification fuite de données
overlap = set(train['post_id']) & set(test['post_id'])
print(f"\n    Fuite train/test : {len(overlap)} posts en commun ✓")

# ------------------------------------------------------------------ #
# 6. Sauvegarde
# ------------------------------------------------------------------ #
df_balanced.to_csv("dataset_clean.csv", index=False, encoding="utf-8-sig")
train.to_csv("train.csv",         index=False, encoding="utf-8-sig")
val.to_csv("val.csv",             index=False, encoding="utf-8-sig")
test.to_csv("test.csv",           index=False, encoding="utf-8-sig")

print(f"\n[6] Fichiers sauvegardés :")
print(f"    ✓ dataset_clean.csv")
print(f"    ✓ train.csv")
print(f"    ✓ val.csv")
print(f"    ✓ test.csv")

# ------------------------------------------------------------------ #
# 7. Visualisations
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Distribution des labels par split", fontsize=13)
colors = ['#e74c3c', '#f39c12', '#2ecc71']

for ax, (split_df, title) in zip(axes, [
    (train, f"Train ({len(train)})"),
    (val,   f"Val ({len(val)})"),
    (test,  f"Test ({len(test)})")
]):
    counts = split_df['label'].value_counts()
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title(title)
    ax.set_ylim(0, target * 1.2)
    for i, (lbl, val_count) in enumerate(counts.items()):
        ax.text(i, val_count + 5, str(val_count), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("distribution_splits.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"\n    ✓ distribution_splits.png")

# Distribution longueur des textes
fig, ax = plt.subplots(figsize=(10, 4))
label_colors = {'trompeur': '#e74c3c', 'non-vérifiable': '#f39c12', 'vrai': '#2ecc71'}
for label, color in label_colors.items():
    subset = df_balanced[df_balanced['label'] == label]['nb_mots']
    ax.hist(subset, bins=40, alpha=0.6, label=label, color=color)
ax.set_xlabel("Nombre de mots")
ax.set_ylabel("Fréquence")
ax.set_title("Distribution de la longueur des textes par classe")
ax.legend()
plt.tight_layout()
plt.savefig("distribution_longueur.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"    ✓ distribution_longueur.png")

print("\n" + "=" * 55)
print("  PHASE 2 TERMINÉE !")
print("  → Lance maintenant : python phase3_baseline.py")
print("=" * 55)
