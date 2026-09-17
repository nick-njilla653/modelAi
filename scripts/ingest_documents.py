"""
GOV-AI 2.0 — Script d'ingestion des documents juridiques.

Usage :
    # Un répertoire entier
    python scripts/ingest_documents.py --dir ./corpus --doc-type loi

    # Un fichier unique, avec le libellé de source qui apparaîtra dans les citations
    python scripts/ingest_documents.py \\
        --file "./Cameroon - Penal Code.pdf" \\
        --source "Code Pénal (Loi n° 2016/007)" \\
        --doc-type loi --force-ocr

Le libellé `--source` est la chaîne que le modèle citera. Par défaut c'est le nom
du fichier : un fichier mal nommé produit donc des citations fausses, d'où
l'intérêt de le donner explicitement.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


async def ingest_files(
    files: list[Path],
    doc_type: str,
    institution: str,
    jurisdiction: str,
    force_ocr: bool,
    dry_run: bool,
    source_label: str | None = None,
) -> int:
    from app.models.schemas import IngestRequest
    from app.services.embedding.embedding_service import get_embedding_service
    from app.services.ingestion.ingestion_service import IngestionService
    from app.storage.elasticsearch_client import ensure_index
    from app.storage.milvus_client import connect_milvus, ensure_collection

    print(f"Documents à traiter : {len(files)}")

    if dry_run:
        for f in files:
            print(f"  [DRY-RUN] {f.name}  ->  source='{source_label or f.name}'")
        return 0

    # Hors application FastAPI, aucune connexion n'est établie au démarrage :
    # sans cela l'insertion Milvus échoue en fin d'ingestion, après l'OCR.
    connect_milvus()
    ensure_collection()
    await ensure_index()

    service = IngestionService()
    # Sans service d'embedding, l'ingestion peuple PostgreSQL mais n'indexe
    # ni Milvus ni Elasticsearch — les documents resteraient introuvables.
    service.set_embedding_service(get_embedding_service())

    success, failed = 0, 0
    for i, file_path in enumerate(files, 1):
        label = source_label or file_path.name
        print(f"[{i}/{len(files)}] {file_path.name} (source='{label}')…", flush=True)
        try:
            request = IngestRequest(
                source=label,
                doc_type=doc_type or None,
                institution=institution or None,
                jurisdiction=jurisdiction or None,
                force_ocr=force_ocr,
            )
            result = await service.ingest_document(
                file_path=str(file_path),
                request=request,
            )
            response = service.to_response(result)
            print(
                f"    OK — {response.chunks_created} chunks, "
                f"langue={response.language_detected}, OCR={response.ocr_used}, "
                f"{response.ingestion_latency_ms / 1000:.1f}s"
            )
            for warning in response.warnings:
                print(f"    avertissement : {warning}")
            success += 1
        except Exception as exc:
            print(f"    ERREUR : {exc}")
            failed += 1

    print(f"\nRésultat : {success} ingéré(s), {failed} échec(s) sur {len(files)}.")
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GOV-AI 2.0 — Ingestion de documents")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dir", help="Répertoire contenant les documents")
    target.add_argument("--file", help="Fichier unique à ingérer")
    parser.add_argument(
        "--source",
        default=None,
        help="Libellé de la source cité dans les réponses (défaut : nom du fichier). "
             "Incompatible avec --dir.",
    )
    parser.add_argument("--doc-type", default="autre", help="Type documentaire (loi, decret, etc.)")
    parser.add_argument("--institution", default="", help="Institution émettrice")
    parser.add_argument("--jurisdiction", default="national", help="Juridiction")
    parser.add_argument("--force-ocr", action="store_true", help="Forcer l'OCR (PDF scanné)")
    parser.add_argument("--dry-run", action="store_true", help="Lister sans ingérer")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dir:
        if args.source:
            print("--source ne s'applique qu'à --file (un libellé par document).")
            sys.exit(2)
        directory = Path(args.dir)
        if not directory.exists():
            print(f"Répertoire introuvable : {directory}")
            sys.exit(1)
        targets = sorted(
            f for f in directory.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        path = Path(args.file)
        if not path.exists():
            print(f"Fichier introuvable : {path}")
            sys.exit(1)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Format non supporté : {path.suffix}")
            sys.exit(1)
        targets = [path]

    sys.exit(asyncio.run(ingest_files(
        files=targets,
        doc_type=args.doc_type,
        institution=args.institution,
        jurisdiction=args.jurisdiction,
        force_ocr=args.force_ocr,
        dry_run=args.dry_run,
        source_label=args.source,
    )))
