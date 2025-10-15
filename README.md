# 🧠 ML for Finance — Project Overview

## 🎯 Objectif du projet
Ce projet vise à **appliquer l’apprentissage automatique (ML)** pour l’estimation et le backtesting de la **Value-at-Risk (VaR)** et de l’**Expected Shortfall (ES)** sur différents portefeuilles financiers (Actions, Crypto, Mixte).  
L’objectif est de comparer les approches classiques, économétriques et ML afin d’identifier les modèles les plus performants selon des critères de robustesse statistique et de conformité Bâle III.

---


---

## 🧮 Méthodologie
1. **Préparation des données** : nettoyage, log-retours, fenêtrage roulant, split train/test.  
2. **Baselines** : Historical Simulation (HS), Variance-Covariance (VC), EWMA.  
3. **Modèles économétriques** : GARCH(1,1)-t.  
4. **Méthodes de simulation** : Monte Carlo (MVN, MVT, Copule t).  
5. **Méthodes ML** : Régression quantile linéaire, GBM quantile.  
6. **Backtesting** : Tests de Kupiec, Christoffersen, Joint CC, zones de Bâle.  
7. **Comparatif** : Classement des modèles selon leurs performances et stabilité.

---

## 📊 Génération automatique du rapport
Une fois toutes les étapes exécutées :

```bash
python make_report_md.py --out REPORT.md
```

👉 Cela scannera automatiquement `./results/` et intégrera :
- Toutes les **figures PNG**
- Tous les **tableaux CSV**
- Un sommaire, un rappel méthodologique et une synthèse automatique

Conversion PDF possible avec :

```bash
pandoc REPORT.md -o REPORT.pdf
```

---

## 📈 Critères de comparaison
- **Coverage test** (Kupiec)  
- **Independence test** (Christoffersen)  
- **Joint test** (CC)  
- **Expected Shortfall moyen**  
- **Classement agrégé** : couverture + stabilité + pertes extrêmes

---

## 🧠 Interprétation générale
- Les modèles **ML (régression quantile, GBM)** captent mieux les queues de distribution.  
- **GARCH‑t** reste robuste sur les portefeuilles volatils.  
- Les **Monte Carlo avec copule t** offrent un compromis théorie / flexibilité.  
- Les **baselines** servent de référence pour mesurer les gains des approches avancées.
