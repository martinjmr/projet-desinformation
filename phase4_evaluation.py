"""
Phase 4 — Évaluation complète du modèle
=======================================
Input  : test.csv, model_tfidf.pkl, vectorizer_tfidf.pkl
Output : evaluation_results.json, evaluation_report.txt

Métriques : Accuracy, Precision, Recall, F1, ROC-AUC, Matrice de confusion
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt

print("=" * 60)
print("  PHASE 4 — Évaluation du modèle")
print("=" * 60)

# ------------------------------------------------------------------ #
# 1. Chargement
# ------------------------------------------------------------------ #
if not os.path.exists("test.csv"):
    print("ERREUR : test.csv introuvable.")
    exit()

if not os.path.exists("model_tfidf.pkl") or not os.path.exists("vectorizer_tfidf.pkl"):
    print("ERREUR : modèle non trouvé. Lance d'abord phase3_train_model.py")
    exit()

test_df = pd.read_csv("test.csv")
model = joblib.load("model_tfidf.pkl")
vectorizer = joblib.load("vectorizer_tfidf.pkl")

print(f"\n[1] Chargement :")
print(f"    Test set : {len(test_df)} posts")
print(f"    Modèle : Logistic Regression + TF-IDF")
print(f"    Classes : {model.classes_.tolist()}")

# ------------------------------------------------------------------ #
# 2. Prédictions
# ------------------------------------------------------------------ #
print(f"\n[2] Prédictions :")

X_test = vectorizer.transform(test_df['texte_clean'])
y_test = test_df['label']

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"    {len(test_df)} prédictions effectuées")

# ------------------------------------------------------------------ #
# 3. Métriques globales
# ------------------------------------------------------------------ #
print(f"\n[3] Résultats globaux :")

accuracy = accuracy_score(y_test, y_pred)
print(f"    Accuracy  : {accuracy:.4f}")

# Weighted average pour multiclass
precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"    Precision (weighted) : {precision_weighted:.4f}")
print(f"    Recall (weighted)    : {recall_weighted:.4f}")
print(f"    F1-Score (weighted)  : {f1_weighted:.4f}")

# ------------------------------------------------------------------ #
# 4. Rapport de classification détaillé
# ------------------------------------------------------------------ #
print(f"\n[4] Rapport par classe :")
print(classification_report(y_test, y_pred, zero_division=0))

# ------------------------------------------------------------------ #
# 5. Matrice de confusion
# ------------------------------------------------------------------ #
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print(f"\n[5] Matrice de confusion :")
print("    Ordre des classes :", model.classes_.tolist())
print(cm)

# ------------------------------------------------------------------ #
# 6. Erreurs d'analyse
# ------------------------------------------------------------------ #
print(f"\n[6] Analyse des erreurs :")

errors_df = test_df.copy()
errors_df['prediction'] = y_pred
errors_df['correct'] = y_pred == y_test

n_errors = (y_pred != y_test).sum()
error_rate = n_errors / len(test_df)

print(f"    Nombre d'erreurs : {n_errors}/{len(test_df)} ({error_rate*100:.1f}%)")

# Erreurs par classe
for label in model.classes_:
    mask = (y_test == label)
    errors_in_class = ((y_pred != y_test) & mask).sum()
    total_in_class = mask.sum()
    if total_in_class > 0:
        error_rate_class = errors_in_class / total_in_class
        print(f"    {label:<15} : {errors_in_class}/{total_in_class} ({error_rate_class*100:.1f}%)")

# ------------------------------------------------------------------ #
# 7. Sauvegarde des résultats
# ------------------------------------------------------------------ #
print(f"\n[7] Sauvegarde :")

evaluation_results = {
    "test_set_size": len(test_df),
    "accuracy": float(accuracy),
    "precision_weighted": float(precision_weighted),
    "recall_weighted": float(recall_weighted),
    "f1_weighted": float(f1_weighted),
    "confusion_matrix": cm.tolist(),
    "classes": model.classes_.tolist(),
    "total_errors": int(n_errors),
    "error_rate": float(error_rate),
}

# Ajouter les métriques par classe
for i, label in enumerate(model.classes_):
    y_test_binary = (y_test == label).astype(int)
    y_pred_binary = (y_pred == label).astype(int)

    evaluation_results[f"precision_{label}"] = float(
        precision_score(y_test_binary, y_pred_binary, zero_division=0)
    )
    evaluation_results[f"recall_{label}"] = float(
        recall_score(y_test_binary, y_pred_binary, zero_division=0)
    )
    evaluation_results[f"f1_{label}"] = float(
        f1_score(y_test_binary, y_pred_binary, zero_division=0)
    )

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

# Rapport texte
with open("evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("  RAPPORT D'ÉVALUATION - DÉTECTION DE DÉSINFORMATION\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Test Set Size : {len(test_df)}\n\n")
    f.write("RÉSULTATS GLOBAUX\n")
    f.write("-" * 60 + "\n")
    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision_weighted:.4f}\n")
    f.write(f"Recall    : {recall_weighted:.4f}\n")
    f.write(f"F1-Score  : {f1_weighted:.4f}\n\n")
    f.write(classification_report(y_test, y_pred, zero_division=0))
    f.write("\n\nMATRICE DE CONFUSION\n")
    f.write("-" * 60 + "\n")
    f.write(str(cm) + "\n")

print(f"    ✓ evaluation_results.json")
print(f"    ✓ evaluation_report.txt")

# ------------------------------------------------------------------ #
# 8. Visualisation
# ------------------------------------------------------------------ #
print(f"\n[8] Visualisations :")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Matrice de confusion
ax = axes[0]
im = ax.imshow(cm, cmap='Blues', aspect='auto')
ax.set_xlabel('Prédiction')
ax.set_ylabel('Réalité')
ax.set_title('Matrice de Confusion')
ax.set_xticks(range(len(model.classes_)))
ax.set_yticks(range(len(model.classes_)))
ax.set_xticklabels(model.classes_, rotation=45, ha='right')
ax.set_yticklabels(model.classes_)

# Ajouter les valeurs dans la matrice
for i in range(len(model.classes_)):
    for j in range(len(model.classes_)):
        text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)

plt.colorbar(im, ax=ax)

# Scores par classe
ax = axes[1]
scores = {label: evaluation_results[f"f1_{label}"] for label in model.classes_}
colors = ['#2ecc71', '#e74c3c', '#f39c12']  # vert, rouge, orange
bars = ax.bar(scores.keys(), scores.values(), color=colors)
ax.set_ylabel('F1-Score')
ax.set_title('F1-Score par classe')
ax.set_ylim([0, 1.0])
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("evaluation_confusion_matrix.png", dpi=150, bbox_inches='tight')
print(f"    ✓ evaluation_confusion_matrix.png")

print("\n" + "=" * 60)
print("  PHASE 4 TERMINÉE !")
print("  → Lance maintenant : python phase5_inference.py")
print("=" * 60)
