# Projet Final – Recommandation d’Investissement via Analyse de Données Multi-Sources

## 🎯 Objectif  
Mettre en place un pipeline automatisé permettant d’agréger quotidiennement des signaux issus de modèles de **clustering**, **classification**, **régression** et **traitement du langage naturel**, afin de générer des **recommandations d’investissement** sur le marché des actions.

Ce projet consolide l’ensemble des Travaux Pratiques (TP) réalisés durant le cours.

---

## 🗂 Contenu du dépôt

- `data/`  
  Contient l’ensemble des données utilisées dans le projet :
  - Fichiers d’actualités quotidiennes  
  - Données de rendements des entreprises  
  - Liste des entreprises étudiées  
  - Ratios financiers associés

- `models/`  
  Contient les modèles développés dans les TP 3, 4 et 5.

- `PosusAI/` et `finbert_finetuned/`  
  Contiennent les modèles de traitement de texte utilisés dans le TP7 (analyse de sentiment).

- `main.py`  
  Script principal qui orchestre l’ensemble du pipeline :
  - Exécution automatique des modules pour chaque entreprise
  - Agrégation des signaux générés
  - Production de recommandations d’investissement

- `main.ipynb`  
  Version Jupyter Notebook du script `main.py` pour une exécution interactive.

- `report.html`  
  Rapport généré automatiquement résumant les résultats et recommandations.

---

## 🚀 Lancer le projet

Pour exécuter l’analyse complète :

- **Option 1** : lancer le script principal
  ```bash
  python main.py
- **Option 2** : Ouvrir et exécuter les cellules du notebook main.ipynb

