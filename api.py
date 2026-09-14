"""
API Flask pour le modèle de détection de désinformation
========================================================

Utilisation :
    python api.py

Puis consultez http://localhost:5000/docs pour la documentation
"""

from flask import Flask, request, jsonify
import joblib
import os
from pathlib import Path
import re

app = Flask(__name__)

# ------------------------------------------------------------------ #
# Chargement du modèle
# ------------------------------------------------------------------ #

def load_model():
    """Charger le modèle et le vectorizer"""
    if not os.path.exists("model_tfidf.pkl") or not os.path.exists("vectorizer_tfidf.pkl"):
        return None, None

    model = joblib.load("model_tfidf.pkl")
    vectorizer = joblib.load("vectorizer_tfidf.pkl")
    return model, vectorizer

MODEL, VECTORIZER = load_model()

if MODEL is None:
    print("⚠️  Modèle non trouvé. Lance d'abord : python phase3_train_model.py")

# ------------------------------------------------------------------ #
# Fonctions utilitaires
# ------------------------------------------------------------------ #

def clean_text(text):
    """Nettoyage du texte"""
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

# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil"""
    return jsonify({
        "name": "Détecteur de Désinformation",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Prédire la classification d'un texte",
            "POST /predict_batch": "Prédire plusieurs textes",
            "GET /health": "Vérifier l'état de l'API",
            "GET /info": "Informations sur le modèle"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Vérifier la santé de l'API"""
    if MODEL is None:
        return jsonify({"status": "error", "message": "Modèle non chargé"}), 503

    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "classes": MODEL.classes_.tolist()
    })

@app.route('/info', methods=['GET'])
def info():
    """Informations sur le modèle"""
    if MODEL is None:
        return jsonify({"status": "error", "message": "Modèle non chargé"}), 503

    return jsonify({
        "model_type": "Logistic Regression + TF-IDF",
        "classes": MODEL.classes_.tolist(),
        "vectorizer_features": VECTORIZER.get_feature_names_out().shape[0],
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédire la classification d'un texte

    Request body:
    {
        "text": "Votre texte à analyser"
    }

    Response:
    {
        "prediction": "trompeur",
        "confidence": 0.92,
        "probabilities": {
            "vrai": 0.05,
            "trompeur": 0.92,
            "non-vérifiable": 0.03
        }
    }
    """
    if MODEL is None:
        return jsonify({"error": "Modèle non chargé"}), 503

    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({"error": "Texte manquant"}), 400

    text = data['text']

    if not text or len(text.strip()) < 5:
        return jsonify({"error": "Texte trop court (minimum 5 caractères)"}), 400

    # Nettoyage et prédiction
    text_clean = clean_text(text)

    if len(text_clean.split()) < 3:
        return jsonify({"error": "Texte trop court après nettoyage"}), 400

    X = VECTORIZER.transform([text_clean])
    prediction = MODEL.predict(X)[0]
    probabilities = MODEL.predict_proba(X)[0]

    # Créer la réponse
    response = {
        "text_preview": text[:100] + "..." if len(text) > 100 else text,
        "prediction": prediction,
        "confidence": float(max(probabilities)),
        "probabilities": {
            label: float(prob)
            for label, prob in zip(MODEL.classes_, probabilities)
        },
        "interpretations": {
            "vrai": "Article fiable basé sur des faits vérifiés",
            "trompeur": "Article contenant de la désinformation",
            "non-vérifiable": "Article difficile à vérifier"
        }
    }

    return jsonify(response)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Prédire plusieurs textes à la fois

    Request body:
    {
        "texts": [
            "Premier texte",
            "Deuxième texte",
            ...
        ]
    }

    Response: List of predictions
    """
    if MODEL is None:
        return jsonify({"error": "Modèle non chargé"}), 503

    data = request.get_json()

    if not data or 'texts' not in data:
        return jsonify({"error": "Liste de textes manquante"}), 400

    texts = data['texts']

    if not isinstance(texts, list):
        return jsonify({"error": "'texts' doit être une liste"}), 400

    if len(texts) == 0:
        return jsonify({"error": "Liste vide"}), 400

    if len(texts) > 100:
        return jsonify({"error": "Maximum 100 textes par requête"}), 400

    # Nettoyage
    texts_clean = [clean_text(t) for t in texts]

    # Prédictions
    X = VECTORIZER.transform(texts_clean)
    predictions = MODEL.predict(X)
    probabilities = MODEL.predict_proba(X)

    # Créer les réponses
    results = []
    for text, pred, probs in zip(texts, predictions, probabilities):
        result = {
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "prediction": pred,
            "confidence": float(max(probs)),
            "probabilities": {
                label: float(prob)
                for label, prob in zip(MODEL.classes_, probs)
            }
        }
        results.append(result)

    return jsonify({"results": results, "total": len(results)})

@app.errorhandler(404)
def not_found(error):
    """Gestion erreur 404"""
    return jsonify({"error": "Endpoint non trouvé"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Gestion erreur 500"""
    return jsonify({"error": "Erreur serveur interne"}), 500

# ------------------------------------------------------------------ #
# Lancement
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    print("=" * 60)
    print("  API de Détection de Désinformation")
    print("=" * 60)

    if MODEL is None:
        print("❌ Modèle non chargé!")
        print("   Lance d'abord : python phase3_train_model.py")
    else:
        print(f"✓ Modèle chargé : {MODEL.__class__.__name__}")
        print(f"✓ Classes : {MODEL.classes_.tolist()}")
        print(f"✓ Vectorizer : {VECTORIZER.get_feature_names_out().shape[0]} features")
        print("\n🚀 Serveur démarré sur http://localhost:5000")
        print("📚 Documentation : http://localhost:5000/docs (avec Swagger UI)")
        print("\nEndpoints :")
        print("  POST /predict       - Prédiction simple")
        print("  POST /predict_batch - Prédictions batch")
        print("  GET  /health        - État du serveur")
        print("  GET  /info          - Info modèle")
        print("=" * 60)

        app.run(debug=True, host='0.0.0.0', port=5000)
