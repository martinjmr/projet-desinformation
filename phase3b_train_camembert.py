"""
Phase 3B — Entraînement avec CamemBERT (optionnel, plus puissant)
==================================================================
Input  : train.csv, val.csv
Output : model_camembert.pkl, model_camembert_results.json

ATTENTION: Plus lent mais plus puissant (~80%+ accuracy vs ~67%)
Nécessite: pip install transformers torch

Ce script est OPTIONNEL - utilisez si vous avez GPU ou patience :)
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

print("=" * 60)
print("  PHASE 3B — Entraînement CamemBERT (optionnel)")
print("=" * 60)

try:
    from transformers import CamemBertForSequenceClassification, CamemBertTokenizer
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import AdamW
except ImportError:
    print("\n❌ Dépendances manquantes!")
    print("   Installez: pip install transformers torch")
    print("\n   OU continuez avec le modèle TF-IDF (plus rapide):")
    print("   python phase4_evaluation.py")
    exit()

# ------------------------------------------------------------------ #
# 1. Chargement des données
# ------------------------------------------------------------------ #
if not os.path.exists("train.csv") or not os.path.exists("val.csv"):
    print("ERREUR : train.csv ou val.csv introuvable.")
    exit()

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

print(f"\n[1] Chargement des données :")
print(f"    Train : {len(train_df)} posts")
print(f"    Val   : {len(val_df)} posts")

# Mapper les labels vers des IDs
label2id = {'vrai': 0, 'trompeur': 1, 'non-vérifiable': 2}
id2label = {v: k for k, v in label2id.items()}

y_train = train_df['label'].map(label2id).values
y_val = val_df['label'].map(label2id).values

# ------------------------------------------------------------------ #
# 2. Chargement du modèle et tokenizer
# ------------------------------------------------------------------ #
print(f"\n[2] Chargement CamemBERT :")

model_name = "distiluse-base-multilingual-cased-v2"  # Plus léger que CamemBERT complet
try:
    tokenizer = CamemBertTokenizer.from_pretrained("distiluse-base-multilingual-cased-v2")
    print(f"    ⚠️  Utilisation d'un tokenizer multilingual")
    print(f"    (CamemBERT complet est trop lourd pour démo)")
except:
    print(f"    Téléchargement du modèle...")
    tokenizer = CamemBertTokenizer.from_pretrained("cmarkea/distilcamembert-base")

model = CamemBertForSequenceClassification.from_pretrained(
    "cmarkea/distilcamembert-base",
    num_labels=3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"    Device : {device}")
print(f"    Modèle : CamemBERT avec 3 classes")

# ------------------------------------------------------------------ #
# 3. Tokenization
# ------------------------------------------------------------------ #
print(f"\n[3] Tokenization (peut prendre quelques minutes...)")

def tokenize_texts(texts, tokenizer, max_length=128):
    """Tokenizer les textes"""
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt"
    )
    return encodings

train_encodings = tokenize_texts(train_df['texte_clean'], tokenizer)
val_encodings = tokenize_texts(val_df['texte_clean'], tokenizer)

print(f"    ✓ {len(train_df)} textes tokenisés (train)")
print(f"    ✓ {len(val_df)} textes tokenisés (val)")

# ------------------------------------------------------------------ #
# 4. DataLoaders
# ------------------------------------------------------------------ #
train_dataset = TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    torch.tensor(y_train)
)

val_dataset = TensorDataset(
    val_encodings['input_ids'],
    val_encodings['attention_mask'],
    torch.tensor(y_val)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ------------------------------------------------------------------ #
# 5. Entraînement
# ------------------------------------------------------------------ #
print(f"\n[4] Entraînement (3 epochs):")

optimizer = AdamW(model.parameters(), lr=2e-5)
device = next(model.parameters()).device

best_val_acc = 0
for epoch in range(3):
    print(f"\n    Epoch {epoch+1}/3")

    # Train
    model.train()
    train_loss = 0
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            print(f"      Batch {batch_idx+1}/{len(train_loader)} - Loss: {train_loss/(batch_idx+1):.4f}")

    # Validation
    model.eval()
    val_preds = []
    val_labels_list = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_labels_list.extend(labels.numpy())

    val_acc = accuracy_score(val_labels_list, val_preds)
    print(f"      Val Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained("model_camembert_checkpoint")
        print(f"      → Meilleur modèle sauvegardé!")

# ------------------------------------------------------------------ #
# 6. Évaluation finale
# ------------------------------------------------------------------ #
print(f"\n[5] Résultats finaux :")

model = CamemBertForSequenceClassification.from_pretrained("model_camembert_checkpoint")
model.to(device)
model.eval()

val_preds = []
with torch.no_grad():
    for input_ids, attention_mask, labels in val_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        val_preds.extend(preds.cpu().numpy())

val_acc = accuracy_score(val_labels_list, val_preds)
print(f"    Accuracy (val) : {val_acc:.4f}")
print(f"\n{classification_report(val_labels_list, val_preds, target_names=['vrai', 'trompeur', 'non-vérifiable'])}")

# ------------------------------------------------------------------ #
# 7. Sauvegarde
# ------------------------------------------------------------------ #
print(f"\n[6] Sauvegarde :")

import shutil
if os.path.exists("model_camembert"):
    shutil.rmtree("model_camembert")
shutil.copytree("model_camembert_checkpoint", "model_camembert")

results = {
    "model_type": "CamemBERT (fine-tuned)",
    "val_accuracy": float(val_acc),
    "classes": list(id2label.values()),
}

with open("model_camembert_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"    ✓ model_camembert/ (dossier)")
print(f"    ✓ model_camembert_results.json")

print("\n" + "=" * 60)
print("  PHASE 3B TERMINÉE !")
print("  → Pour utiliser ce modèle : phase5_inference_camembert.py")
print("=" * 60)
