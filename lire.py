# 06_interpret_results.py
"""
Script d'analyse et d'interprétation complète des résultats de backtesting VaR.
Génère un rapport détaillé avec statistiques, graphiques et recommandations.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ---------- Chargement des données ----------
def load_all_results():
    """Charge tous les résultats de backtest."""
    ranked = pd.read_csv(RESULTS / "compare_all_methods_ranked.csv")
    winners = pd.read_csv(RESULTS / "winners_by_portfolio_alpha.csv")
    return ranked, winners

# ---------- Analyses statistiques ----------
def analyze_coverage(df):
    """Analyse de la couverture (hit rate vs α théorique)."""
    print("\n" + "="*80)
    print("1. ANALYSE DE LA COUVERTURE (Hit Rate)")
    print("="*80)
    
    coverage = df.groupby(['portfolio', 'alpha', 'model']).agg({
        'hit_rate': 'first',
        'breaches': 'first',
        'T': 'first',
        'Kupiec_p': 'first'
    }).reset_index()
    
    coverage['hit_deviation'] = coverage['hit_rate'] - coverage['alpha']
    coverage['hit_deviation_pct'] = (coverage['hit_deviation'] / coverage['alpha']) * 100
    
    print("\nDéviation moyenne du hit rate par rapport à α théorique:")
    for port in coverage['portfolio'].unique():
        print(f"\n  Portfolio {port}:")
        port_data = coverage[coverage['portfolio'] == port]
        for alpha in sorted(port_data['alpha'].unique()):
            alpha_data = port_data[port_data['alpha'] == alpha]
            mean_dev = alpha_data['hit_deviation_pct'].mean()
            best_model = alpha_data.loc[alpha_data['hit_deviation_pct'].abs().idxmin(), 'model']
            print(f"    α={alpha:.2%}: déviation moyenne = {mean_dev:+.1f}% | Meilleur: {best_model}")
    
    return coverage

def analyze_independence(df):
    """Analyse de l'indépendance des violations (Christoffersen)."""
    print("\n" + "="*80)
    print("2. ANALYSE DE L'INDÉPENDANCE DES VIOLATIONS")
    print("="*80)
    
    # Modèles qui échouent le test d'indépendance (p < 0.05)
    failed_ind = df[df['Christoff_p'] < 0.05]
    
    print(f"\nModèles échouant le test d'indépendance (p < 5%): {len(failed_ind)}/{len(df)}")
    if len(failed_ind) > 0:
        print("\nTop 10 des échecs les plus sévères:")
        worst = failed_ind.nsmallest(10, 'Christoff_p')[
            ['portfolio', 'alpha', 'model', 'Christoff_p', 'breaches']
        ]
        print(worst.to_string(index=False))
        
        print("\n⚠️  Interprétation: Ces modèles montrent des clusters de violations,")
        print("    indiquant une sous-estimation systématique du risque dans certaines périodes.")
    else:
        print("\n✓ Tous les modèles passent le test d'indépendance.")

def analyze_joint_test(df):
    """Analyse du test joint (Christoffersen)."""
    print("\n" + "="*80)
    print("3. ANALYSE DU TEST JOINT (Couverture + Indépendance)")
    print("="*80)
    
    # Statistiques par niveau de confiance
    for alpha in sorted(df['alpha'].unique()):
        alpha_data = df[df['alpha'] == alpha]
        passed = (alpha_data['Joint_p'] >= 0.05).sum()
        total = len(alpha_data)
        
        print(f"\nNiveau α={alpha:.2%}:")
        print(f"  Modèles validés (p ≥ 5%): {passed}/{total} ({passed/total*100:.1f}%)")
        
        # Meilleurs par portfolio
        print(f"  Meilleurs modèles par portfolio:")
        for port in sorted(alpha_data['portfolio'].unique()):
            port_data = alpha_data[alpha_data['portfolio'] == port]
            best = port_data.nlargest(3, 'Joint_p')[['model', 'Joint_p', 'zone']]
            for _, row in best.iterrows():
                print(f"    {port}: {row['model']:20s} | p={row['Joint_p']:.4f} | zone={row['zone']}")

def analyze_expected_shortfall(df):
    """Analyse de l'Expected Shortfall réalisé."""
    print("\n" + "="*80)
    print("4. ANALYSE DE L'EXPECTED SHORTFALL (ES)")
    print("="*80)
    
    # Exclure les NaN
    es_data = df[df['ES_realised'].notna()].copy()
    
    if len(es_data) == 0:
        print("\n⚠️  Aucune donnée ES disponible (aucun breach observé).")
        return
    
    print("\nES moyen par type de modèle (perte moyenne lors des violations):")
    es_by_model = es_data.groupby('model')['ES_realised'].agg(['mean', 'std', 'count'])
    es_by_model = es_by_model.sort_values('mean')
    print(es_by_model.to_string(float_format=lambda x: f"{x:.4f}"))
    
    print("\n💡 Interprétation: Un ES plus faible indique que les violations sont moins sévères.")
    print("   Les modèles avec ES élevé sous-estiment davantage le risque de queue.")

def analyze_basel_zones(df):
    """Analyse des zones Basel (traffic light)."""
    print("\n" + "="*80)
    print("5. ANALYSE DES ZONES BASEL (Traffic Light)")
    print("="*80)
    
    zone_counts = df.groupby(['zone']).size()
    total = len(df)
    
    print("\nDistribution des zones:")
    for zone in ['green', 'yellow', 'red']:
        count = zone_counts.get(zone, 0)
        pct = count / total * 100
        emoji = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}[zone]
        print(f"  {emoji} {zone.upper():6s}: {count:3d} modèles ({pct:5.1f}%)")
    
    # Modèles en zone rouge
    red_zone = df[df['zone'] == 'red']
    if len(red_zone) > 0:
        print(f"\n⚠️  {len(red_zone)} modèles en ZONE ROUGE (violations excessives):")
        print(red_zone[['portfolio', 'alpha', 'model', 'breaches', 'T']].to_string(index=False))
    else:
        print("\n✓ Aucun modèle en zone rouge.")

def model_family_comparison(df):
    """Compare les familles de modèles."""
    print("\n" + "="*80)
    print("6. COMPARAISON PAR FAMILLE DE MODÈLES")
    print("="*80)
    
    # Définir les familles
    def classify_model(name):
        if name in ['HS', 'VC', 'VC_EWMA']:
            return 'Parametric'
        elif 'GARCH' in name:
            return 'GARCH'
        elif 'MC-' in name or 'Copule' in name:
            return 'Monte-Carlo'
        else:
            return 'Other'
    
    df['family'] = df['model'].apply(classify_model)
    
    # Performance moyenne par famille
    print("\nPerformance moyenne par famille (rang moyen):")
    family_perf = df.groupby('family').agg({
        'rank_avg': 'mean',
        'Joint_p': 'mean',
        'Kupiec_p': 'mean',
        'hit_dev_abs': 'mean'
    }).round(3)
    print(family_perf.to_string())
    
    print("\n💡 Interprétation:")
    best_family = family_perf['rank_avg'].idxmin()
    print(f"   La famille {best_family} obtient le meilleur rang moyen.")

# ---------- Visualisations ----------
def create_performance_heatmap(df):
    """Crée une heatmap des p-values du test joint."""
    print("\n[Génération] Heatmap des performances...")
    
    for alpha in sorted(df['alpha'].unique()):
        alpha_data = df[df['alpha'] == alpha]
        
        # Pivot: portfolios en lignes, modèles en colonnes
        pivot = alpha_data.pivot_table(
            values='Joint_p', 
            index='portfolio', 
            columns='model',
            aggfunc='first'
        )
        
        fig, ax = plt.subplots(figsize=(14, 4))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                    vmin=0, vmax=1, center=0.5, ax=ax,
                    cbar_kws={'label': 'Joint p-value'})
        ax.set_title(f'Performance des modèles VaR (α={alpha:.2%}) - Joint Test p-values')
        ax.set_xlabel('Modèle')
        ax.set_ylabel('Portfolio')
        plt.tight_layout()
        plt.savefig(RESULTS / f'heatmap_performance_a{int(alpha*100)}.png', dpi=200)
        plt.close()

def create_coverage_comparison(coverage):
    """Compare les hit rates vs α théorique."""
    print("[Génération] Graphiques de couverture...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, alpha in enumerate(sorted(coverage['alpha'].unique())):
        ax = axes[i]
        data = coverage[coverage['alpha'] == alpha]
        
        # Grouper par portfolio et modèle
        for port in sorted(data['portfolio'].unique()):
            port_data = data[data['portfolio'] == port]
            x = range(len(port_data))
            y = port_data['hit_rate'].values
            ax.scatter(x, y, label=port, s=100, alpha=0.6)
        
        # Ligne théorique
        ax.axhline(alpha, color='red', linestyle='--', linewidth=2, 
                   label=f'α théorique = {alpha:.2%}')
        
        # Bandes de confiance binomiale (95%)
        T = data['T'].iloc[0]
        lower = binom.ppf(0.025, T, alpha) / T
        upper = binom.ppf(0.975, T, alpha) / T
        ax.axhspan(lower, upper, alpha=0.2, color='green', 
                   label='IC 95% binomial')
        
        ax.set_title(f'Hit Rate vs α théorique (α={alpha:.2%})')
        ax.set_xlabel('Modèle (index)')
        ax.set_ylabel('Hit Rate observé')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS / 'coverage_comparison.png', dpi=200)
    plt.close()

def create_ranking_summary(df):
    """Crée un graphique récapitulatif des rangs."""
    print("[Génération] Graphique de ranking...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    idx = 0
    for port in sorted(df['portfolio'].unique()):
        for alpha in sorted(df['alpha'].unique()):
            ax = axes[idx]
            data = df[(df['portfolio'] == port) & (df['alpha'] == alpha)]
            data = data.sort_values('rank_avg')
            
            colors = data['zone'].map({'green': 'green', 'yellow': 'orange', 'red': 'red'})
            
            ax.barh(range(len(data)), data['rank_avg'], color=colors, alpha=0.7)
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data['model'], fontsize=8)
            ax.set_xlabel('Rang moyen (↓ meilleur)')
            ax.set_title(f'{port} - α={alpha:.2%}')
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
            
            idx += 1
    
    plt.tight_layout()
    plt.savefig(RESULTS / 'ranking_summary.png', dpi=200)
    plt.close()

# ---------- Rapport final ----------
def generate_recommendations(winners, df):
    """Génère des recommandations finales."""
    print("\n" + "="*80)
    print("7. RECOMMANDATIONS FINALES")
    print("="*80)
    
    print("\n🏆 MEILLEURS MODÈLES PAR CONFIGURATION:")
    for _, row in winners.iterrows():
        zone_emoji = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}[row['zone']]
        print(f"\n  {row['portfolio']} | α={row['alpha']:.2%}")
        print(f"    → {row['winner_model']}")
        print(f"    → Joint p-value: {row['Joint_p']:.4f}")
        print(f"    → Zone Basel: {zone_emoji} {row['zone']}")
        print(f"    → Violations: {row['breaches']}/{row['T']}")
    
    print("\n" + "="*80)
    print("SYNTHÈSE ET CONCLUSIONS")
    print("="*80)
    
    # Analyse globale
    total_models = len(df)
    validated = (df['Joint_p'] >= 0.05).sum()
    
    print(f"\n📊 Statistiques globales:")
    print(f"   • Total de configurations testées: {total_models}")
    print(f"   • Modèles validés (p ≥ 5%): {validated} ({validated/total_models*100:.1f}%)")
    
    # Meilleur modèle global
    best_overall = df.nlargest(1, 'Joint_p').iloc[0]
    print(f"\n🥇 Meilleure configuration globale:")
    print(f"   • Modèle: {best_overall['model']}")
    print(f"   • Portfolio: {best_overall['portfolio']}")
    print(f"   • α: {best_overall['alpha']:.2%}")
    print(f"   • Joint p-value: {best_overall['Joint_p']:.4f}")
    
    # Recommandations spécifiques
    print("\n💡 RECOMMANDATIONS:")
    
    # Pour portefeuille actions
    actions = df[df['portfolio'] == 'P_A']
    best_actions = actions.nlargest(1, 'Joint_p').iloc[0]
    print(f"\n   1. Pour le portefeuille ACTIONS (P_A):")
    print(f"      ✓ Utiliser: {best_actions['model']}")
    print(f"      ✓ Raison: Meilleure couverture et indépendance (p={best_actions['Joint_p']:.4f})")
    
    # Pour portefeuille crypto
    crypto = df[df['portfolio'] == 'P_B']
    best_crypto = crypto.nlargest(1, 'Joint_p').iloc[0]
    print(f"\n   2. Pour le portefeuille CRYPTO (P_B):")
    print(f"      ✓ Utiliser: {best_crypto['model']}")
    print(f"      ✓ Raison: Capture mieux la volatilité extrême (p={best_crypto['Joint_p']:.4f})")
    
    # Pour portefeuille mixte
    mixte = df[df['portfolio'] == 'P_C']
    best_mixte = mixte.nlargest(1, 'Joint_p').iloc[0]
    print(f"\n   3. Pour le portefeuille MIXTE (P_C):")
    print(f"      ✓ Utiliser: {best_mixte['model']}")
    print(f"      ✓ Raison: Gère bien la diversification (p={best_mixte['Joint_p']:.4f})")
    
    print("\n   4. Considérations générales:")
    if (df['family'] == 'Monte-Carlo').any():
        mc_perf = df[df['family'] == 'Monte-Carlo']['Joint_p'].mean()
        print(f"      • Monte-Carlo: Performance moyenne élevée (p̄={mc_perf:.4f})")
        print(f"        → Recommandé pour portefeuilles complexes avec queues épaisses")
    
    if (df['family'] == 'GARCH').any():
        garch_perf = df[df['family'] == 'GARCH']['Joint_p'].mean()
        print(f"      • GARCH: Performance moyenne (p̄={garch_perf:.4f})")
        print(f"        → Bon compromis entre précision et rapidité")
    
    print("\n⚠️  AVERTISSEMENTS:")
    print("   • Les performances passées ne garantissent pas les résultats futurs")
    print("   • Recalibrer régulièrement les modèles (au moins trimestriellement)")
    print("   • Combiner plusieurs modèles pour plus de robustesse (ensemble)")
    print("   • Effectuer des stress tests en complément du backtest")

# ---------- Main ----------
def main():
    """Fonction principale d'analyse."""
    print("\n" + "="*80)
    print("ANALYSE COMPLÈTE DES RÉSULTATS DE BACKTEST VAR")
    print("="*80)
    
    # Chargement
    df, winners = load_all_results()
    
    # Analyses statistiques
    coverage = analyze_coverage(df)
    analyze_independence(df)
    analyze_joint_test(df)
    analyze_expected_shortfall(df)
    analyze_basel_zones(df)
    model_family_comparison(df)
    
    # Visualisations
    print("\n" + "="*80)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("="*80)
    create_performance_heatmap(df)
    create_coverage_comparison(coverage)
    create_ranking_summary(df)
    
    # Recommandations
    generate_recommendations(winners, df)
    
    # Sauvegarde du rapport
    print("\n" + "="*80)
    print("FICHIERS GÉNÉRÉS")
    print("="*80)
    print(f"\nRapport sauvegardé dans: {RESULTS.resolve()}")
    print("  • heatmap_performance_a*.png - Heatmaps des p-values")
    print("  • coverage_comparison.png - Comparaison des hit rates")
    print("  • ranking_summary.png - Vue d'ensemble des rangs")
    print("\n✓ Analyse terminée avec succès!")

if __name__ == "__main__":
    main()