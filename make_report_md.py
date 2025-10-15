#!/usr/bin/env python3
"""
make_report_md.py
-----------------
Génère un rapport Markdown (REPORT.md) dans le répertoire courant en
incluant automatiquement toutes les figures et tableaux de résultats
trouvés dans ./results/.

Usage:
    python make_report_md.py [--out REPORT.md]

Options:
    --out : nom du fichier de sortie (défaut: REPORT.md)

Pré-requis:
    - Avoir exécuté les scripts d'estimation/backtest qui écrivent dans ./results/
    - pandas installé (pour lire les CSV et générer les tableaux Markdown).
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse
import re
import pandas as pd
import numpy as np
import textwrap

# ---------- Utils ----------

def df_to_markdown_table(df: pd.DataFrame, index=False) -> str:
    """Convertit un DataFrame en table Markdown simple (sans dépendance tabulate)."""
    if df is None or df.empty:
        return "_Aucune donnée disponible._"
    df = df.copy()
    if not index:
        df = df.reset_index(drop=True)

    # Convert to strings
    cols = list(df.columns)
    rows = [[str(x) for x in row] for row in df.values.tolist()]

    # Determine column widths
    col_widths = [len(str(c)) for c in cols]
    for row in rows:
        for j, cell in enumerate(row):
            if len(cell) > col_widths[j]:
                col_widths[j] = len(cell)

    # Build header
    header = "| " + " | ".join(str(c).ljust(col_widths[j]) for j, c in enumerate(cols)) + " |"
    sep    = "| " + " | ".join("-" * col_widths[j]           for j, _ in enumerate(cols)) + " |"
    body_lines = []
    for row in rows:
        body_lines.append("| " + " | ".join(str(cell).ljust(col_widths[j]) for j, cell in enumerate(row)) + " |")
    return "\n".join([header, sep] + body_lines)

def natural_key(s: str):
    """Clé de tri 'naturel' pour classer plot_P_A_a1.png avant plot_P_A_a10.png."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def list_images(results: Path) -> dict[str, list[str]]:
    """Retourne un dict {section: [paths]} groupant les figures par famille."""
    imgs = sorted([p for p in results.glob("*.png")], key=lambda p: natural_key(p.name))
    groups = {
        "Baselines": [],
        "GARCH": [],
        "Monte-Carlo": [],
        "ML - Régression quantile": [],
        "ML - GBM quantile": [],
        "Comparatif (traffic-light)": [],
        "Autres": [],
    }
    for p in imgs:
        n = p.name
        if re.search(r"_QR_a\d+\.png$", n):
            groups["ML - Régression quantile"].append(n)
        elif re.search(r"_GBM_a\d+\.png$", n):
            groups["ML - GBM quantile"].append(n)
        elif re.search(r"_GARCH_.*_a\d+\.png$", n):
            groups["GARCH"].append(n)
        elif re.search(r"_MC_a\d+\.png$", n):
            groups["Monte-Carlo"].append(n)
        elif re.search(r"^compare_.*_a\d+\.png$", n):
            groups["Comparatif (traffic-light)"].append(n)
        elif re.search(r"^plot_.*_a\d+\.png$", n):
            # Étape 2 baselines (VC/HS/EWMA)
            groups["Baselines"].append(n)
        else:
            groups["Autres"].append(n)
    return groups

def try_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        pass
    return None

def add_section_images(md_lines: list[str], title: str, images: list[str]):
    if not images: 
        return
    md_lines.append(f"### {title}")
    for img in images:
        md_lines.append(f"![{img}](results/{img})")
    md_lines.append("")

def add_section_csv(md_lines: list[str], title: str, df: pd.DataFrame | None, note: str | None = None):
    if df is None or df.empty:
        return
    md_lines.append(f"### {title}")
    if note:
        md_lines.append(note)
    # Limit large numeric precision for readability
    df_show = df.copy()
    for c in df_show.columns:
        if pd.api.types.is_numeric_dtype(df_show[c]):
            df_show[c] = df_show[c].map(lambda x: f"{x:.6g}" if pd.notnull(x) else "")
    md_lines.append(df_to_markdown_table(df_show, index=False))
    md_lines.append("")

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="REPORT.md", help="Nom du fichier Markdown de sortie")
    args = parser.parse_args()

    root = Path(".").resolve()
    results = root / "results"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Header
    md = []
    md.append("# ML for Finance — Rapport final")
    md.append("")
    md.append(f"**Généré automatiquement le :** {now}")
    md.append("")
    md.append("> Ce rapport scanne `./results/` et inclut automatiquement les figures et tableaux disponibles.")
    md.append("")
    md.append("## Sommaire")
    md.append("- [Résumé](#résumé)")
    md.append("- [Méthodologie (rappel)](#méthodologie-rappel)")
    md.append("- [Figures](#figures)")
    md.append("- [Tableaux de backtest](#tableaux-de-backtest)")
    md.append("- [Comparatif & traffic-light](#comparatif--traffic-light)")
    md.append("- [Conclusion rapide](#conclusion-rapide)")
    md.append("")

    # Résumé
    md.append("## Résumé")
    md.append(textwrap.dedent("""\
    Ce document récapitule les résultats d'estimation et de backtesting de la VaR (1 % / 5 %)
    pour trois portefeuilles (Actions, Crypto, Mixte) en comparant : baselines (Historique, VC/EWMA),
    GARCH(1,1)-t, approches Monte-Carlo (MVN, MVT, copule t) et ML (régression quantile et GBM-quantile).
    """).strip())
    md.append("")

    # Méthodologie (rappel bref)
    md.append("## Méthodologie (rappel)")
    md.append("- **Fenêtre roulante** : estimation → prévision J+1 → avance d'1 jour.")
    md.append("- **Tests** : Kupiec (couverture), Christoffersen (indépendance), Joint (CC), ES réalisé.")
    md.append("- **Zone Bâle** : vert/jaune/rouge selon nb d'exceptions vs quantiles binomiaux.")
    md.append("")

    # Figures
    md.append("## Figures")
    groups = list_images(results)
    add_section_images(md, "Baselines (HS, VC, EWMA)", groups.get("Baselines", []))
    add_section_images(md, "GARCH(1,1) — innovations t", groups.get("GARCH", []))
    add_section_images(md, "Monte-Carlo — MVN / MVT / Copule t", groups.get("Monte-Carlo", []))
    add_section_images(md, "ML — Régression quantile", groups.get("ML - Régression quantile", []))
    add_section_images(md, "ML — GBM quantile", groups.get("ML - GBM quantile", []))
    add_section_images(md, "Comparatif (traffic-light)", groups.get("Comparatif (traffic-light)", []))
    add_section_images(md, "Autres", groups.get("Autres", []))

    # Tableaux de backtest
    md.append("## Tableaux de backtest")
    df2 = try_read_csv(results / "backtest_summary.csv")
    df3 = try_read_csv(results / "backtest_summary_GARCH.csv")
    df4 = try_read_csv(results / "backtest_summary_MC.csv")
    df5 = try_read_csv(results / "backtest_summary_QR.csv")
    df6 = try_read_csv(results / "backtest_summary_GBM.csv")

    add_section_csv(md, "Baselines (HS/VC/EWMA)", df2)
    add_section_csv(md, "GARCH(1,1)-t", df3)
    add_section_csv(md, "Monte-Carlo (MVN/MVT/Copule t)", df4)
    add_section_csv(md, "ML — Régression quantile", df5)
    add_section_csv(md, "ML — GBM quantile", df6)

    # Comparatif
    md.append("## Comparatif & traffic-light")
    winners = try_read_csv(results / "winners_by_portfolio_alpha.csv")
    if winners is not None and not winners.empty:
        add_section_csv(md, "Gagnants par (portefeuille, α)", winners, note=None)
    else:
        md.append("_Fichier `winners_by_portfolio_alpha.csv` non trouvé. Exécutez l'étape 5 pour le générer._\n")

    comp_raw = try_read_csv(results / "compare_all_methods_raw.csv")
    comp_ranked = try_read_csv(results / "compare_all_methods_ranked.csv")
    add_section_csv(md, "Comparatif — brut", comp_raw)
    add_section_csv(md, "Comparatif — classé", comp_ranked)

    # Conclusion
    md.append("## Conclusion rapide")
    md.append("- Privilégier les modèles validés par la p-value **conjointe** et un **ES** faible.")
    md.append("- En pratique, conserver 2–3 modèles \"champions\" (ex. GARCH‑t, Copule‑t, ML‑quantile) et surveiller leur robustesse temporelle.")
    md.append("")

    out = root / (args.out if args.out.endswith(".md") else f"{args.out}.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Rapport Markdown écrit : {out}")
    if not results.exists():
        print("[WARN] Le dossier ./results/ n'existe pas encore. Exécutez d'abord vos scripts de génération.")
    else:
        if not any(results.glob("*.png")) and not any(results.glob("*.csv")):
            print("[WARN] Aucun .png ou .csv trouvé dans ./results/. Le rapport contiendra les sections vides.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
