# Détecteur de Désinformation 🛡️

Un système complet de machine learning pour détecter et classifier les contenus en ligne comme **vrais**, **trompeurs** ou **non-vérifiables**.

## 📊 Pipeline du Projet

```
Phase 1: Collecte de données
    ↓
Phase 2: Pré-traitement & nettoyage
    ↓
Phase 3: Entraînement du modèle
    ↓
Phase 4: Évaluation
    ↓
Phase 5: Prédictions sur de nouvelles données
```

## 🚀 Démarrage Rapide

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 2. Préparer les données

Si vous avez déjà un CSV brut (`posts_raw.csv`), lancez le pré-traitement :

```bash
python phase2_pretraitement.py
```

Cela générera :
- `dataset_clean.csv` - dataset nettoyé
- `train.csv`, `val.csv`, `test.csv` - splits de données

### 3. Entraîner le modèle

```bash
python phase3_train_model.py
```

Génère :
- `model_tfidf.pkl` - modèle entraîné
- `vectorizer_tfidf.pkl` - vectorizer TF-IDF
- `model_results.json` - résultats d'entraînement

### 4. Évaluer les performances

```bash
python phase4_evaluation.py
```

Génère :
- `evaluation_results.json` - métriques détaillées
- `evaluation_report.txt` - rapport textuel
- `evaluation_confusion_matrix.png` - visualisations

### 5. Faire des prédictions

```bash
# Sur un texte unique
python phase5_inference.py "La Terre est plate selon une nouvelle étude"

# Sur un fichier (une ligne = un texte)
python phase5_inference.py --file articles_a_verifier.txt
```

## 📈 Architecture du Modèle

**Approche actuelle :** TF-IDF + Logistic Regression
- **Vectorisation** : TF-IDF avec 5000 features max
- **N-grams** : Unigrammes + bigrammes
- **Stop words** : Suppression des stop words français
- **Classification** : Logistic Regression avec équilibrage des classes

### Performances

- **Accuracy** : ~X% (voir `evaluation_results.json`)
- **Precision/Recall** : Équilibré par classe
- **Classes** :
  - ✅ `vrai` : Contenu fiable et vérifiable
  - ❌ `trompeur` : Contenu contenant de la désinformation
  - ❓ `non-vérifiable` : Contenu difficile à vérifier

## 📁 Structure des Fichiers

```
projet-desinformation/
├── collect_reddit_noapikey.py      # Phase 1: Collecte Reddit
├── phase2_pretraitement.py         # Phase 2: Nettoyage des données
├── phase3_train_model.py           # Phase 3: Entraînement
├── phase4_evaluation.py            # Phase 4: Évaluation
├── phase5_inference.py             # Phase 5: Prédictions
├── requirements.txt                # Dépendances Python
├── dataset_clean.csv               # Dataset final (tous les splits)
├── train.csv, val.csv, test.csv    # Données d'entraînement/test
├── model_tfidf.pkl                 # Modèle entraîné
├── vectorizer_tfidf.pkl            # Vectorizer TF-IDF
├── model_results.json              # Résultats d'entraînement
├── evaluation_results.json         # Résultats d'évaluation
└── README.md                       # Ce fichier
```

## 🔧 Utilisation Avancée

### Format d'entrée pour prédictions en batch

Créez un fichier `articles.txt` avec un article par ligne :

```
Première news à vérifier ici
Deuxième news sur un sujet différent
Troisième news avec plusieurs mots clés
```

Puis lancez :
```bash
python phase5_inference.py --file articles.txt
```

Génère : `articles_predictions.json`

### Résultats de prédiction

Format JSON :
```json
{
  "texte": "La Terre est plate...",
  "prediction": "trompeur",
  "confiance": 0.92,
  "probabilites": {
    "vrai": 0.05,
    "trompeur": 0.92,
    "non-vérifiable": 0.03
  }
}
```

## 📊 Dataset

**Source** : Posts Reddit collectés manuellement avec fact-checks
- **Taille** : ~X posts (voir logs d'exécution)
- **Classes** : 3 (vrai, trompeur, non-vérifiable)
- **Langage** : Français
- **Distribution** : Équilibrée après rééquilibrage

### Collecte supplémentaire

Pour ajouter plus de données :

```bash
python collect_reddit_noapikey.py
```

Nécessite : `factchecks_reddit.csv` (not provided - à compléter)

## 🎯 Améliorations Possibles

1. **Modèles plus puissants** :
   - BERT fine-tuning (transformers)
   - CamemBERT pour le français
   - FastText avec n-grams

2. **Features supplémentaires** :
   - Longueur du texte
   - Présence de nombres/dates
   - URLs externes
   - Scores de sentiment

3. **Données** :
   - Augmentation du dataset
   - Balancing entre classes plus fin
   - Validation croisée k-fold

4. **Productionnalisation** :
   - API Flask/FastAPI
   - Containerisation Docker
   - CI/CD Pipeline
   - Monitoring des performances

## 📝 Licence

MIT License

## 👨‍💻 Auteur

Martin Jomier

---

**Besoin d'aide ?** Consultez les logs d'exécution ou modifiez les paramètres dans chaque script.
