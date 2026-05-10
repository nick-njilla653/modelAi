"""
GOV-AI 2.0 — Runner d'évaluation (étude ablative B0→B4).
Usage : python eval/run_eval.py --baseline B4 --dataset eval/datasets/qa_bilingual_annotated.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def run(baseline_id: str, dataset_path: str, k_values: list[int]) -> None:
    from app.services.evaluation.evaluation_service import EvaluationService

    print(f"\n=== GOV-AI 2.0 — Évaluation {baseline_id} ===")
    print(f"Dataset : {dataset_path}")
    print(f"K values : {k_values}\n")

    svc = EvaluationService()

    try:
        report = await svc.run_evaluation(
            dataset_path=dataset_path,
            baseline_id=baseline_id,
            k_values=k_values,
        )
    except FileNotFoundError as exc:
        print(f"ERREUR : {exc}")
        sys.exit(1)

    data = report.to_dict()

    print("── Métriques Retrieval ──────────────────────────────────")
    ret = data["retrieval"]
    for k in k_values:
        print(f"  P@{k}    = {ret['precision_at_k'].get(str(k), ret['precision_at_k'].get(k, 'N/A')):.4f}")
    for k in k_values:
        print(f"  nDCG@{k} = {ret['ndcg_at_k'].get(str(k), ret['ndcg_at_k'].get(k, 'N/A')):.4f}")
    print(f"  MRR     = {ret['mrr']:.4f}")

    print("\n── Métriques Génération ─────────────────────────────────")
    gen = data["generation"]
    print(f"  Faithfulness      = {gen['faithfulness']:.4f}")
    print(f"  Citation Precision= {gen['citation_precision']:.4f}  (target ≥ 0.95)")
    print(f"  Hallucination Rate= {gen['hallucination_rate']:.4f}  (target ≤ 0.05)")
    print(f"  ISB               = {gen['isb']:.4f}  (target ≥ 0.85)")

    print("\n── Métriques Système ────────────────────────────────────")
    sys_m = data["system"]["end_to_end"]
    print(f"  p50 = {sys_m['p50_ms']:.1f} ms  (target ≤ 5000 ms)")
    print(f"  p95 = {sys_m['p95_ms']:.1f} ms  (target ≤ 15000 ms)")

    print("\n── Contraintes du mémoire ───────────────────────────────")
    for constraint, met in data["constraints_met"].items():
        status = "✓ OK" if met else "✗ NON RESPECTÉE"
        print(f"  {constraint}: {status}")

    print(f"\nRapport sauvegardé : voir eval/reports/")
    print("=" * 55)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GOV-AI 2.0 — Évaluation")
    parser.add_argument(
        "--baseline", default="B4",
        choices=["B0", "B1", "B2", "B3", "B4"],
        help="Baseline à évaluer",
    )
    parser.add_argument(
        "--dataset",
        default="eval/datasets/qa_bilingual_annotated.json",
        help="Chemin vers le dataset annoté",
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=[1, 3, 5, 10],
        help="Valeurs de k pour les métriques",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.baseline, args.dataset, args.k))
