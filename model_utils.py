"""
Utilitaires réutilisables pour le projet de détection de désinformation
=========================================================================
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------ #
# Nettoyage de texte
# ------------------------------------------------------------------ #

def clean_text(text):
    """
    Nettoyage standardisé des textes.
    Applique la même logique que phase2_pretraitement.py
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)              # URLs
    text = re.sub(r'u/\w+|r/\w+', '', text)                   # Mentions Reddit
    text = re.sub(r'\[removed\]|\[deleted\]', '', text)        # Posts supprimés
    text = re.sub(
        r'[^\w\s\'\-àâçéèêëîïôùûüÿœæÀÂÇÉÈÊËÎÏÔÙÛÜŸŒÆ.,!?;:]',
        ' ', text
    )
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_text_stats(text):
    """Récupérer les statistiques d'un texte"""
    clean = clean_text(text)
    words = clean.split()

    return {
        "original_length": len(text),
        "clean_length": len(clean),
        "word_count": len(words),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "unique_words": len(set(words)),
    }


# ------------------------------------------------------------------ #
# Évaluation
# ------------------------------------------------------------------ #

def print_detailed_results(y_true, y_pred, y_pred_proba=None, classes=None):
    """
    Afficher un rapport détaillé d'évaluation
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, classification_report)

    if classes is None:
        classes = np.unique(y_true)

    print("\n" + "="*60)
    print("RAPPORT D'ÉVALUATION")
    print("="*60)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy global : {accuracy:.4f}")

    print("\n" + classification_report(y_true, y_pred, labels=classes))

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print("Matrice de confusion :")
    print(cm)

    return {"accuracy": accuracy, "confusion_matrix": cm}


# ------------------------------------------------------------------ #
# Données
# ------------------------------------------------------------------ #

def load_dataset(filename):
    """Charger un CSV du projet"""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"{filename} non trouvé")

    df = pd.read_csv(path)
    print(f"✓ {filename} chargé : {len(df)} lignes, {len(df.columns)} colonnes")

    return df


def get_class_distribution(df, label_column="label"):
    """Obtenir la distribution des classes"""
    dist = df[label_column].value_counts()
    total = len(df)

    print("\nDistribution des classes :")
    for label, count in dist.items():
        pct = 100 * count / total
        print(f"  {label:<15} : {count:>5} ({pct:>5.1f}%)")

    return dist


def check_data_leakage(df1, df2, column="post_id"):
    """Vérifier les fuites entre deux datasets"""
    if column not in df1.columns or column not in df2.columns:
        print(f"⚠ Colonne '{column}' manquante")
        return None

    overlap = set(df1[column]) & set(df2[column])

    if len(overlap) > 0:
        print(f"⚠ {len(overlap)} entrées en commun entre les datasets!")
        return overlap
    else:
        print(f"✓ Pas de fuite de données")
        return None


# ------------------------------------------------------------------ #
# Visualisation
# ------------------------------------------------------------------ #

def create_class_distribution_plot(y_true, classes=None, filename=None):
    """Créer un graphique de distribution des classes"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non disponible")
        return

    if classes is None:
        classes = np.unique(y_true)

    counts = [np.sum(y_true == c) for c in classes]
    colors = ['#2ecc71', '#e74c3c', '#f39c12'][:len(classes)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(classes, counts, color=colors)
    ax.set_ylabel("Nombre d'exemples")
    ax.set_title("Distribution des classes")

    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé : {filename}")

    plt.show()


# ------------------------------------------------------------------ #
# Prédictions
# ------------------------------------------------------------------ #

def predict_batch(texts, model, vectorizer, classes=None):
    """
    Faire des prédictions sur un batch de textes
    """
    cleaned_texts = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned_texts)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    if classes is None:
        classes = model.classes_

    results = []
    for text, pred, probs in zip(texts, predictions, probabilities):
        result = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "prediction": pred,
            "confidence": float(max(probs)),
            "probabilities": {c: float(p) for c, p in zip(classes, probs)}
        }
        results.append(result)

    return results


def save_predictions_json(results, filename):
    """Sauvegarder les prédictions en JSON"""
    import json

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✓ Prédictions sauvegardées : {filename}")


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # Test nettoyage
    test_text = "Bonjour! Voici un URL https://example.com et u/reddit"
    print("Texte original :", test_text)
    print("Texte nettoyé :", clean_text(test_text))

    # Test stats
    stats = get_text_stats(test_text)
    print("\nStatistiques :", stats)
