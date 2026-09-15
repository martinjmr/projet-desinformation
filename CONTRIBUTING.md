# Guide de Contribution

Merci de votre intérêt pour ce projet! Voici comment contribuer.

## 🚀 Comment commencer

1. **Fork** le repository
2. **Clone** votre fork:
   ```bash
   git clone https://github.com/VOTRE-USERNAME/projet-desinformation.git
   ```
3. **Créez une branche** pour votre feature:
   ```bash
   git checkout -b feature/ma-feature
   ```

## 📝 Avant de commencer

- Installez les dépendances: `pip install -r requirements.txt`
- Lisez le `README.md` pour comprendre l'architecture

## 🔧 Types de contributions bienvenues

### 🐛 Corrections de bugs
- Créez une issue décrivant le bug
- Proposez une PR avec la correction

### ✨ Nouvelles features
- IA/ML improvements (meilleurs modèles, features)
- Optimisations de performance
- Documentation
- Tests

### 📚 Améliorations

**Modèles ML:**
- Essayez BERT, CamemBERT (cf. `phase3b_train_camembert.py`)
- Ajoutez des features (sentiment, longueur texte, etc.)
- Améliorez le data balancing

**Code:**
- Refactoring pour meilleure lisibilité
- Optimisations
- Meilleur error handling

**Documentation:**
- Exemples supplémentaires
- Explications détaillées
- Guides de troubleshooting

## 📋 Checklist avant de soumettre une PR

- [ ] Code testé localement
- [ ] Pas d'erreurs de syntaxe
- [ ] Commentaires pertinents ajoutés
- [ ] README mis à jour (si nécessaire)
- [ ] Commit messages clairs et en français
- [ ] Pas de fichiers volumineux (`.pkl`, `.csv` > 1MB)

## 💬 Communication

- **Issues**: Pour rapporter des bugs ou proposer des features
- **Discussions**: Pour les questions générales
- **Pull Requests**: Pour les contributions de code

## 📖 Structure du projet

```
projet-desinformation/
├── phase1: Collecte de données
├── phase2: Pré-traitement
├── phase3: Entraînement TF-IDF
├── phase3b: Entraînement CamemBERT (optionnel)
├── phase4: Évaluation
├── phase5: Prédictions
└── api.py: Interface production
```

## ⚖️ Licence

Ce projet est sous licence MIT. En contribuant, vous acceptez que votre code soit utilisé sous cette licence.

---

Merci pour votre contribution! 🙏
