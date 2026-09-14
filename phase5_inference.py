"""
Phase 5 — Prédictions sur nouvelles données
============================================

Utilisation :
    python phase5_inference.py "Ma première news à vérifier"
    python phase5_inference.py --file test_articles.txt
"""

import sys
import joblib
import os
import json
from pathlib import Path

print("=" * 60)
print("  PHASE 5 — Prédictions sur de nouvelles données")
print("=" * 60)

# ------------------------------------------------------------------ #
# 1. Chargement du modèle
# ------------------------------------------------------------------ #
if not os.path.exists("model_tfidf.pkl") or not os.path.exists("vectorizer_tfidf.pkl"):
    print("ERREUR : modèle non trouvé.")
    print("Lance d'abord : python phase3_train_model.py")
    exit()

model = joblib.load("model_tfidf.pkl")
vectorizer = joblib.load("vectorizer_tfidf.pkl")

print(f"\n[1] Modèle chargé :")
print(f"    Classes : {model.classes_.tolist()}")

# ------------------------------------------------------------------ #
# 2. Fonction de prédiction
# ------------------------------------------------------------------ #
def clean_text(text):
    """Nettoyage du texte (même que en phase 2)"""
    import re
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'u/\w+|r/\w+', '', text)
    text = re.sub(r'\[removed\]|\[deleted\]', '', text)
    text = re.sub(
        r'[^\w\s\'\-àâçéèêëîïôùûüÿœæÀÂÇÉÈÊËÎÏÔÙÛÜŸŒÆ.,!?;:]',
        ' ', text
    )
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_news(text):
    """Prédire la classification d'une news"""
    text_clean = clean_text(text)

    if not text_clean or len(text_clean.split()) < 5:
        return {
            "texte": text[:100] + "..." if len(text) > 100 else text,
            "erreur": "Texte trop court ou invalide (minimum 5 mots après nettoyage)"
        }

    X = vectorizer.transform([text_clean])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    # Créer un dictionnaire des probabilités
    probs_dict = {label: float(prob)
                  for label, prob in zip(model.classes_, probabilities)}

    # Trouver la confiance (probabilité la plus haute)
    confidence = float(max(probabilities))

    return {
        "texte": text[:100] + "..." if len(text) > 100 else text,
        "prediction": prediction,
        "confiance": confidence,
        "probabilites": probs_dict,
        "interpretations": {
            "vrai": "Article fiable basé sur des faits vérifiés",
            "trompeur": "Article contenant de la désinformation ou des fausses affirmations",
            "non-vérifiable": "Article difficile à vérifier ou contenant des affirmations non justifiées"
        }
    }

# ------------------------------------------------------------------ #
# 3. Interface utilisateur
# ------------------------------------------------------------------ #

def main():
    if len(sys.argv) < 2:
        print(f"\n[2] Usage :")
        print(f"    python phase5_inference.py \"Votre texte à vérifier\"")
        print(f"    python phase5_inference.py --file fichier.txt")
        print(f"\nExemple :")
        result = predict_news("La terre est plate")
        print_result(result)
        return

    if sys.argv[1] == "--file":
        if len(sys.argv) < 3:
            print("ERREUR : spécifiez un fichier")
            return

        file_path = sys.argv[2]
        if not os.path.exists(file_path):
            print(f"ERREUR : fichier '{file_path}' introuvable")
            return

        print(f"\n[2] Lecture du fichier : {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"    {len(texts)} textes trouvés\n")

        results = []
        for i, text in enumerate(texts, 1):
            result = predict_news(text)
            result['numero'] = i
            results.append(result)
            print_result(result)
            print()

        # Sauvegarde en JSON
        output_file = Path(file_path).stem + "_predictions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Résultats sauvegardés : {output_file}")

    else:
        # Texte passé en argument
        text = " ".join(sys.argv[1:])
        print(f"\n[2] Texte analysé :")
        result = predict_news(text)
        print_result(result)

def print_result(result):
    """Afficher les résultats de manière lisible"""
    if "erreur" in result:
        print(f"  ✗ {result['erreur']}")
        return

    print(f"  Texte : {result['texte']}")
    print(f"  Prédiction : {result['prediction'].upper()}")
    print(f"  Confiance : {result['confiance']*100:.1f}%")
    print(f"  Interprétation : {result['interpretations'][result['prediction']]}")
    print(f"  Probabilités :")
    for label, prob in result['probabilites'].items():
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"    {label:<15} : {bar} {prob:.1%}")

if __name__ == "__main__":
    main()

    print("\n" + "=" * 60)
    print("  Pour des prédictions en batch, utilisez --file")
    print("=" * 60)
