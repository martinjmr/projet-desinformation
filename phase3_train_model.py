"""
Phase 3 — Entraînement du modèle de détection de désinformation
=================================================================
Input  : train.csv, val.csv
Output : model_tfidf.pkl, vectorizer_tfidf.pkl, model_results.json

Approches disponibles :
1. TF-IDF + Logistic Regression (rapide, baseline)
2. BERT fine-tuning (plus puissant mais plus lent)
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

print("=" * 60)
print("  PHASE 3 — Entraînement du modèle")
print("=" * 60)

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
print(f"    Distribution train : {train_df['label'].value_counts().to_dict()}")

# ------------------------------------------------------------------ #
# 2. Vectorisation TF-IDF
# ------------------------------------------------------------------ #
print(f"\n[2] Vectorisation TF-IDF :")

vectorizer = TfidfVectorizer(
    max_features=5000,           # Garder les 5000 features les plus importantes
    ngram_range=(1, 2),          # Unigrammes + bigrammes
    min_df=2,                    # Ignorer les termes dans <2 documents
    max_df=0.8,                  # Ignorer les termes dans >80% des documents
    lowercase=True,
    stop_words='french'          # Supprimer les stop words français
)

X_train = vectorizer.fit_transform(train_df['texte_clean'])
X_val = vectorizer.transform(val_df['texte_clean'])

print(f"    Vocabulaire : {X_train.shape[1]} features")
print(f"    X_train : {X_train.shape}")
print(f"    X_val   : {X_val.shape}")

# ------------------------------------------------------------------ #
# 3. Entraînement Logistic Regression
# ------------------------------------------------------------------ #
print(f"\n[3] Entraînement Logistic Regression :")

y_train = train_df['label']
y_val = val_df['label']

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',     # Compenser le déséquilibre de classes
    n_jobs=-1                    # Utiliser tous les CPU
)

model.fit(X_train, y_train)
print(f"    Modèle entraîné ✓")

# ------------------------------------------------------------------ #
# 4. Évaluation
# ------------------------------------------------------------------ #
print(f"\n[4] Évaluation :")

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"    Accuracy (train) : {train_acc:.4f}")
print(f"    Accuracy (val)   : {val_acc:.4f}")

print(f"\n    Classification Report (Validation) :")
print(classification_report(y_val, y_val_pred))

# Matrice de confusion
cm = confusion_matrix(y_val, y_val_pred)
print(f"    Matrice de confusion :")
print(cm)

# Sauvegarde des résultats
results = {
    "train_accuracy": float(train_acc),
    "val_accuracy": float(val_acc),
    "classes": model.classes_.tolist(),
    "confusion_matrix": cm.tolist(),
}

# Ajouter les métriques par classe
for label in model.classes_:
    y_val_label = (y_val == label).astype(int)
    y_pred_label = (y_val_pred == label).astype(int)

    results[f"precision_{label}"] = float(precision_score(y_val_label, y_pred_label))
    results[f"recall_{label}"] = float(recall_score(y_val_label, y_pred_label))
    results[f"f1_{label}"] = float(f1_score(y_val_label, y_pred_label))

# ------------------------------------------------------------------ #
# 5. Sauvegarde du modèle
# ------------------------------------------------------------------ #
print(f"\n[5] Sauvegarde :")

joblib.dump(model, "model_tfidf.pkl")
joblib.dump(vectorizer, "vectorizer_tfidf.pkl")

with open("model_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"    ✓ model_tfidf.pkl")
print(f"    ✓ vectorizer_tfidf.pkl")
print(f"    ✓ model_results.json")

# ------------------------------------------------------------------ #
# 6. Feature importance
# ------------------------------------------------------------------ #
print(f"\n[6] Features importantes (top 20) :")

# Récupérer les coefficients du modèle
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_

# Pour chaque classe, afficher les 10 features avec les plus hauts coefficients
for i, class_label in enumerate(model.classes_):
    print(f"\n    Classe : '{class_label}'")
    top_indices = np.argsort(coefficients[i])[-10:][::-1]
    for idx in top_indices:
        print(f"      • {feature_names[idx]:<20} ({coefficients[i][idx]:+.4f})")

print("\n" + "=" * 60)
print("  PHASE 3 TERMINÉE !")
print("  → Lance maintenant : python phase4_evaluation.py")
print("=" * 60)
