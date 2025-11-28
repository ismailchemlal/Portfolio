"""
VaR Géopolitique 2.0 - Script de Démarrage Rapide
===================================================

Ce script permet de tester rapidement le modèle avec des paramètres prédéfinis.
Idéal pour une première démonstration ou validation.

Auteur: CHEMLAL Ismail
"""

import sys
import os
from datetime import datetime

# Bannière ASCII
BANNER = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██╗   ██╗ █████╗ ██████╗      ██████╗ ███████╗ ██████╗                ║
║   ██║   ██║██╔══██╗██╔══██╗    ██╔════╝ ██╔════╝██╔═══██╗               ║
║   ██║   ██║███████║██████╔╝    ██║  ███╗█████╗  ██║   ██║               ║
║   ╚██╗ ██╔╝██╔══██║██╔══██╗    ██║   ██║██╔══╝  ██║   ██║               ║
║    ╚████╔╝ ██║  ██║██║  ██║    ╚██████╔╝███████╗╚██████╔╝               ║
║     ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═╝     ╚═════╝ ╚══════╝ ╚═════╝                ║
║                                                                           ║
║              GÉOPOLITIQUE 2.0 - Quick Start Demo                         ║
║         Modèle Prédictif de Risques par Intelligence Géopolitique        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

def print_banner():
    """Affiche la bannière du projet"""
    print("\033[96m" + BANNER + "\033[0m")
    print(f"\n📅 Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python version: {sys.version.split()[0]}")
    print("="*79)

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    print("\n🔍 Vérification des dépendances...")
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'yfinance': 'yfinance',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib'
    }
    
    missing = []
    installed = []
    
    for display_name, package_name in required_packages.items():
        try:
            __import__(package_name)
            installed.append(display_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            missing.append(package_name)
            print(f"  ✗ {display_name}")
    
    if missing:
        print("\n⚠️  Packages manquants détectés!")
        print(f"   Exécutez: pip install {' '.join(missing)}")
        print("\n   Ou installez tout via: pip install -r requirements.txt")
        return False
    
    print(f"\n✅ Toutes les dépendances sont installées ({len(installed)}/{len(required_packages)})")
    return True

def interactive_menu():
    """Menu interactif pour choisir le mode d'exécution"""
    print("\n" + "="*79)
    print("📋 MENU PRINCIPAL")
    print("="*79)
    print("\n🎯 Choisissez un mode d'exécution:\n")
    print("  1. 🚀 Demo Rapide (S&P 500, 3 ans)")
    print("  2. 📊 Analyse Complète (S&P 500, 5 ans)")
    print("  3. 🌍 Multi-Indices (US, EU, Asia)")
    print("  4. ⚙️  Mode Personnalisé")
    print("  5. 📖 Afficher la Documentation")
    print("  6. ❌ Quitter")
    
    while True:
        try:
            choice = input("\n👉 Votre choix (1-6): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6']:
                return choice
            else:
                print("⚠️  Choix invalide. Veuillez entrer un nombre entre 1 et 6.")
        except KeyboardInterrupt:
            print("\n\n👋 Interruption détectée. Au revoir!")
            sys.exit(0)

def run_quick_demo():
    """Mode 1: Démonstration rapide"""
    print("\n" + "="*79)
    print("🚀 MODE: DEMO RAPIDE")
    print("="*79)
    print("\nConfiguration:")
    print("  • Indice: S&P 500 (^GSPC)")
    print("  • Période: 3 ans")
    print("  • Niveau de confiance: 95%")
    print("\n⏳ Démarrage de l'analyse...\n")
    
    try:
        from geopolitical_var_model import GeopoliticalVaRModel
        
        model = GeopoliticalVaRModel(confidence_level=0.95, window=252)
        results = model.run_complete_analysis(ticker='^GSPC', years=3)
        
        print("\n" + "="*79)
        print("✨ RÉSUMÉ DES RÉSULTATS")
        print("="*79)
        
        improvement = results['improvement']
        n_obs = results['results_geopolitical']['n_observations']
        
        print(f"\n📊 Données analysées: {n_obs} jours de trading")
        print(f"🎯 Amélioration du modèle: {improvement:.1f}%")
        print(f"✅ Test de Kupiec: {results['results_geopolitical']['kupiec_result']}")
        
        print("\n💡 Recommandation:")
        if improvement > 20:
            print("   Le modèle géopolitique apporte une amélioration SIGNIFICATIVE!")
        elif improvement > 10:
            print("   Le modèle géopolitique améliore modérément les prévisions.")
        else:
            print("   Les deux modèles ont des performances comparables.")
        
        return True
        
    except ImportError:
        print("\n❌ ERREUR: geopolitical_var_model.py non trouvé!")
        print("   Assurez-vous que le fichier est dans le même répertoire.")
        return False
    except Exception as e:
        print(f"\n❌ ERREUR lors de l'exécution: {str(e)}")
        return False

def run_full_analysis():
    """Mode 2: Analyse complète"""
    print("\n" + "="*79)
    print("📊 MODE: ANALYSE COMPLÈTE")
    print("="*79)
    print("\nConfiguration:")
    print("  • Indice: S&P 500 (^GSPC)")
    print("  • Période: 5 ans")
    print("  • Niveau de confiance: 95%")
    print("  • Exports: CSV + Graphiques")
    print("\n⏳ Démarrage de l'analyse complète (cela peut prendre 1-2 minutes)...\n")
    
    try:
        from geopolitical_var_model import GeopoliticalVaRModel
        
        model = GeopoliticalVaRModel(confidence_level=0.95, window=252)
        results = model.run_complete_analysis(ticker='^GSPC', years=5)
        
        # Vérifier les fichiers exportés
        print("\n📁 Fichiers générés:")
        if os.path.exists('var_geopolitical_results.csv'):
            print("  ✓ var_geopolitical_results.csv")
        if os.path.exists('var_geopolitical_analysis.png'):
            print("  ✓ var_geopolitical_analysis.png")
        
        print("\n💾 Utilisez ces fichiers pour des analyses supplémentaires!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        return False

def run_multi_indices():
    """Mode 3: Analyse multi-indices"""
    print("\n" + "="*79)
    print("🌍 MODE: ANALYSE MULTI-INDICES")
    print("="*79)
    
    indices = {
        'S&P 500 (US)': '^GSPC',
        'EUROSTOXX 50 (EU)': '^STOXX50E',
        'Nikkei 225 (Japan)': '^N225'
    }
    
    print("\nIndices à analyser:")
    for name in indices.keys():
        print(f"  • {name}")
    
    print("\n⏳ Démarrage des analyses (cela peut prendre 3-5 minutes)...\n")
    
    try:
        from geopolitical_var_model import GeopoliticalVaRModel
        
        results_all = {}
        
        for name, ticker in indices.items():
            print(f"\n📊 Analyse: {name}")
            print("-" * 79)
            
            model = GeopoliticalVaRModel(confidence_level=0.95, window=252)
            results_all[name] = model.run_complete_analysis(ticker=ticker, years=3)
        
        # Résumé comparatif
        print("\n" + "="*79)
        print("📊 COMPARAISON MULTI-INDICES")
        print("="*79)
        print(f"\n{'Indice':<20} {'Amélioration':<15} {'Test Kupiec':<15}")
        print("-" * 79)
        
        for name in indices.keys():
            improvement = results_all[name]['improvement']
            kupiec = results_all[name]['results_geopolitical']['kupiec_result']
            print(f"{name:<20} {improvement:>6.1f}%{'':<8} {kupiec:<15}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        return False

def run_custom_mode():
    """Mode 4: Configuration personnalisée"""
    print("\n" + "="*79)
    print("⚙️  MODE: PERSONNALISÉ")
    print("="*79)
    
    try:
        # Saisie du ticker
        print("\n📈 Quel indice souhaitez-vous analyser?")
        print("   Exemples: ^GSPC (S&P 500), ^STOXX50E (EuroStoxx), ^N225 (Nikkei)")
        ticker = input("   Ticker: ").strip().upper()
        if not ticker:
            ticker = '^GSPC'
            print(f"   → Utilisation de la valeur par défaut: {ticker}")
        
        # Saisie de la période
        print("\n📅 Nombre d'années d'historique?")
        years_input = input("   Années (1-10, défaut=5): ").strip()
        years = int(years_input) if years_input.isdigit() and 1 <= int(years_input) <= 10 else 5
        print(f"   → Période sélectionnée: {years} ans")
        
        # Saisie du niveau de confiance
        print("\n🎯 Niveau de confiance pour la VaR?")
        conf_input = input("   Confiance (90, 95, 99, défaut=95): ").strip()
        if conf_input in ['90', '95', '99']:
            confidence = int(conf_input) / 100
        else:
            confidence = 0.95
        print(f"   → Niveau de confiance: {confidence*100:.0f}%")
        
        # Confirmation
        print("\n" + "-"*79)
        print("✅ Configuration:")
        print(f"   • Ticker: {ticker}")
        print(f"   • Période: {years} ans")
        print(f"   • Confiance: {confidence*100:.0f}%")
        print("-"*79)
        
        confirm = input("\n▶️  Lancer l'analyse? (O/n): ").strip().lower()
        if confirm == 'n':
            print("❌ Analyse annulée.")
            return False
        
        print("\n⏳ Démarrage de l'analyse personnalisée...\n")
        
        from geopolitical_var_model import GeopoliticalVaRModel
        
        model = GeopoliticalVaRModel(confidence_level=confidence, window=252)
        results = model.run_complete_analysis(ticker=ticker, years=years)
        
        print("\n✅ Analyse personnalisée terminée avec succès!")
        return True
        
    except ValueError:
        print("\n❌ ERREUR: Valeur invalide saisie.")
        return False
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        return False

def show_documentation():
    """Mode 5: Afficher la documentation"""
    print("\n" + "="*79)
    print("📖 DOCUMENTATION RAPIDE")
    print("="*79)
    
    doc = """
╔═══════════════════════════════════════════════════════════════════════════╗
║ QU'EST-CE QUE LA VaR (VALUE-AT-RISK) ?                                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ La VaR est une mesure statistique qui estime la perte maximale probable  ║
║ d'un portefeuille sur une période donnée avec un niveau de confiance     ║
║ spécifique.                                                               ║
║                                                                           ║
║ Exemple: VaR 95% = 2.5%                                                  ║
║ → Il y a 5% de chance de perdre plus de 2.5% du capital.                ║
╚═══════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════╗
║ POURQUOI UN MODÈLE GÉOPOLITIQUE ?                                        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Les modèles VaR traditionnels échouent lors de crises géopolitiques car: ║
║                                                                           ║
║  ❌ Ils supposent la stabilité des paramètres                            ║
║  ❌ Ils ignorent les signaux d'alerte précoce                            ║
║  ❌ Ils sous-estiment les événements extrêmes                            ║
║                                                                           ║
║ Notre modèle intègre:                                                     ║
║  ✅ GPR Index (tensions géopolitiques)                                   ║
║  ✅ Régimes de marché dynamiques                                         ║
║  ✅ Adaptation temps réel aux crises                                     ║
╚═══════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════╗
║ FICHIERS GÉNÉRÉS                                                          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  📄 var_geopolitical_results.csv                                         ║
║     → Données complètes: prix, rendements, VaR, régimes                  ║
║                                                                           ║
║  📊 var_geopolitical_analysis.png                                        ║
║     → Graphiques: GPR, VaR comparison, régimes, performance              ║
║                                                                           ║
║  📈 backtest_metrics.csv                                     ║
║     → Métriques détaillées des deux modèles                              ║
╚═══════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════╗
║ INTERPRÉTATION DES RÉSULTATS                                             ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Test de Kupiec:                                                           ║
║   ✅ ACCEPTÉ → Le modèle est statistiquement valide                      ║
║   ❌ REJETÉ  → Taux de violation trop élevé, modèle imprécis            ║
║                                                                           ║
║ Amélioration:                                                             ║
║   > 20% → Amélioration SIGNIFICATIVE                                     ║
║   10-20% → Amélioration MODÉRÉE                                          ║
║   < 10% → Amélioration FAIBLE                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

Pour plus d'informations, consultez README.md
    """
    
    print(doc)
    input("\n📖 Appuyez sur Entrée pour revenir au menu...")

def main():
    """Point d'entrée principal"""
    print_banner()
    
    # Vérification des dépendances
    if not check_dependencies():
        print("\n❌ Installation requise avant de continuer.")
        sys.exit(1)
    
    # Boucle principale du menu
    while True:
        choice = interactive_menu()
        
        if choice == '1':
            success = run_quick_demo()
            if success:
                print("\n✅ Demo rapide terminée avec succès!")
        
        elif choice == '2':
            success = run_full_analysis()
            if success:
                print("\n✅ Analyse complète terminée avec succès!")
        
        elif choice == '3':
            success = run_multi_indices()
            if success:
                print("\n✅ Analyse multi-indices terminée avec succès!")
        
        elif choice == '4':
            run_custom_mode()
        
        elif choice == '5':
            show_documentation()
            continue  # Retour direct au menu
        
        elif choice == '6':
            print("\n" + "="*79)
            print("👋 Merci d'avoir utilisé VaR Géopolitique 2.0!")
            print("="*79)
            print("\n📚 Pour aller plus loin:")
            print("  • Consultez README.md pour la documentation complète")
            print("  • Explorez var_geopolitique_analysis.ipynb pour l'analyse interactive")
            print("  • Visitez le repository GitHub pour les mises à jour")
            print("\n💡 N'oubliez pas de ⭐ le projet si vous l'avez trouvé utile!")
            print("\n🔬 Développé par CHEMLAL Ismail - 2025")
            print("="*79 + "\n")
            sys.exit(0)
        
        # Proposer de continuer ou quitter
        print("\n" + "-"*79)
        continue_choice = input("▶️  Voulez-vous effectuer une autre analyse? (O/n): ").strip().lower()
        if continue_choice == 'n':
            print("\n👋 Au revoir!")
            sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Programme interrompu par l'utilisateur.")
        print("👋 Au revoir!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {str(e)}")
        print("\n🐛 Si le problème persiste, veuillez:")
        print("  1. Vérifier que toutes les dépendances sont installées")
        print("  2. Consulter la documentation dans README.md")
        print("  3. Ouvrir une issue sur GitHub avec le message d'erreur")
        sys.exit(1)
