"""
GOV-AI 2.0 — Runner d'évaluation (étude ablative B0→B4).

Usage :
    python eval/run_eval.py --baseline B4
    python eval/run_eval.py --baseline B0 B1 B2 B3      # recherche seule, rapide
    python eval/run_eval.py --baseline all              # B0 à B4, puis tableau comparatif

B0 à B3 ne génèrent aucune réponse : quelques secondes par question. B4 passe par
le système complet, soit plusieurs minutes par question sur le matériel actuel.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ALL = ["B0", "B1", "B2", "B3", "B4"]


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/d"


def _print_report(data: dict, k_values: list[int]) -> None:
    print(f"\n=== {data['baseline_id']} — {data['baseline_description']} ===")
    print(f"  {data['num_queries']} questions · mode {data['mode']}")

    ret = data["retrieval"]
    print("  Récupération")
    print("    " + "  ".join(f"HR@{k}={_fmt(ret['hit_rate_at_k'].get(k))}" for k in k_values))
    print("    " + "  ".join(f"nDCG@{k}={_fmt(ret['ndcg_at_k'].get(k))}" for k in k_values))
    print(f"    MRR={_fmt(ret['mrr'])}")

    gen = data.get("generation")
    if gen:
        print("  Génération")
        print(f"    Rappel des mots-clés attendus = {_fmt(gen['keyword_recall'])}")
        print(f"    Fidélité (lexicale)           = {_fmt(gen['faithfulness'])}")
        print(f"    Précision des citations       = {_fmt(gen['citation_precision'])}  (cible ≥ 0.95)")
        print(f"    Taux de refus                 = {_fmt(gen['refusal_rate'])}")
        print(f"    ISB                           = {_fmt(gen['isb'])}  (cible ≥ 0.85)")

    system = data["system"]
    latency = system.get("end_to_end") or system.get("retrieval")
    label = "bout en bout" if "end_to_end" in system else "recherche"
    print(f"  Latence ({label}) : p50={latency['p50_ms']:.0f} ms  p95={latency['p95_ms']:.0f} ms"
          f"  · taux d'erreur={_fmt(system['error_rate'])}")

    for constraint, met in data["constraints_met"].items():
        print(f"    {'✓' if met else '✗'} {constraint}")


async def run(baselines: list[str], dataset_path: str, k_values: list[int]) -> None:
    from app.services.evaluation.evaluation_service import EvaluationService

    svc = EvaluationService()
    reports = []
    for baseline_id in baselines:
        try:
            report = await svc.run_evaluation(
                dataset_path=dataset_path, baseline_id=baseline_id, k_values=k_values,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERREUR : {exc}")
            sys.exit(1)
        reports.append(report)
        _print_report(report.to_dict(), k_values)

    if len(reports) > 1:
        path = svc.save_ablation_summary(reports)
        print(f"\nTableau comparatif : {path}")
    print("Rapports individuels : eval/reports/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GOV-AI 2.0 — Évaluation")
    parser.add_argument(
        "--baseline", nargs="+", default=["B4"], choices=ALL + ["all"],
        help="Baseline(s) à évaluer, ou « all »",
    )
    parser.add_argument(
        "--dataset", default="eval/datasets/qa_bilingual_annotated.json",
        help="Chemin vers le jeu annoté",
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=[1, 3, 5, 10],
        help="Valeurs de k pour les métriques",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected = ALL if "all" in args.baseline else args.baseline
    asyncio.run(run(selected, args.dataset, args.k))
