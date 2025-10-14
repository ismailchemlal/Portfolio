# 🚀 VaR Géopolitique 2.0 - Guide Démarrage Rapide (5 minutes)

## ⚡ Installation Express

```bash
# 1. Cloner le projet
git clone https://github.com/votre-username/var-geopolitique.git
cd var-geopolitique

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer le script
python quick_start.py
```

**C'est tout !** Le menu interactif se lance automatiquement.

---

## 📊 Options du Menu

### Option 1: Demo Rapide (⏱️ ~30 secondes)
- Analyse S&P 500 sur 3 ans
- Parfait pour première découverte
- Affiche amélioration vs modèle traditionnel

### Option 2: Analyse Complète (⏱️ ~1-2 minutes)
- Analyse S&P 500 sur 5 ans
- Génère CSV + graphiques
- Backtesting rigoureux complet

### Option 3: Multi-Indices (⏱️ ~3-5 minutes)
- US (S&P 500) + EU (EUROSTOXX) + Asia (Nikkei)
- Comparaison performance par région
- Tableau récapitulatif

### Option 4: Personnalisé
- Choisir votre ticker (ex: ^FTSE, ^DJI)
- Définir période d'analyse
- Sélectionner niveau de confiance (90%, 95%, 99%)

---

## 📁 Fichiers Générés

Après exécution, vous trouverez:

```
var-geopolitique/
├── var_geopolitical_results.csv       ← Données complètes
├── var_geopolitical_analysis.png      ← Graphiques
└── backtest_metrics.csv (optionnel)   ← Métriques détaillées
```

---

## 🎯 Exemple de Résultat

```
════════════════════════════════════════════════════════════════
📊 RÉSULTATS BACKTESTING
════════════════════════════════════════════════════════════════
Métrique                  Traditionnel    Géopolitique
────────────────────────────────────────────────────────────────
Violations                18              7
Taux violation           15.0%           5.8%
Test de Kupiec           ❌ REJETÉ       ✅ ACCEPTÉ
────────────────────────────────────────────────────────────────

✨ AMÉLIORATION: 61% réduction des violations
```

---

## 🐍 Utilisation Programmatique

```python
from geopolitical_var_model import GeopoliticalVaRModel

# Initialiser
model = GeopoliticalVaRModel(confidence_level=0.95)

# Analyser
results = model.run_complete_analysis(ticker='^GSPC', years=5)

# Accéder aux résultats
data = results['data']  # DataFrame complet
improvement = results['improvement']  # Amélioration en %
```

---

## 📖 Dashboard Interactif (Optionnel)

Si vous avez installé Streamlit:

```bash
streamlit run dashboard.py
```

Ouvre dans votre navigateur un dashboard avec:
- 📊 Visualisations temps réel
- 🎯 Probabilités des régimes
- ⚠️ Alertes géopolitiques
- 📈 Métriques de performance

---

## 🔧 Dépannage Rapide

### Erreur "Module not found"
```bash
pip install numpy pandas yfinance scipy matplotlib
```

### Erreur de téléchargement Yahoo Finance
- Vérifiez votre connexion internet
- Essayez un autre ticker (ex: ^DJI au lieu de ^GSPC)

### Graphiques ne s'affichent pas
```bash
pip install --upgrade matplotlib
```

---

## 📚 Aller Plus Loin

- **Documentation complète**: Lire `README.md`
- **Analyse interactive**: Ouvrir `var_geopolitique_analysis.ipynb`
- **Code source**: Consulter `geopolitical_var_model.py`
- **Présentation**: Voir `PRESENTATION.md`

---

## 💡 Comprendre les Résultats

### Test de Kupiec
- ✅ **ACCEPTÉ**: Le modèle est statistiquement valide
- ❌ **REJETÉ**: Taux de violation trop élevé

### Amélioration
- **> 20%**: Amélioration SIGNIFICATIVE 🎉
- **10-20%**: Amélioration MODÉRÉE ✅
- **< 10%**: Amélioration FAIBLE ⚠️

### Violations
- Nombre de fois où perte réelle > VaR prédite
- Plus c'est bas, meilleur est le modèle

---

## 🆘 Support

- 🐛 **Bugs**: [Ouvrir une issue](https://github.com/votre-username/var-geopolitique/issues)
- 💬 **Questions**: [Discussions](https://github.com/votre-username/var-geopolitique/discussions)
- 📧 **Contact**: votre.email@example.com

---

## ⭐ Vous aimez le projet?

N'oubliez pas de lui donner une étoile sur GitHub !

**Développé par CHEMLAL Ismail** | 2025