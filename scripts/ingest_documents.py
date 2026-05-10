"""
GOV-AI 2.0 — Script d'ingestion en masse des documents juridiques.
Usage : python scripts/ingest_documents.py --dir ./corpus --doc-type loi
"""
import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def ingest_directory(
    directory: Path,
    doc_type: str,
    institution: str,
    jurisdiction: str,
    force_ocr: bool,
    dry_run: bool,
) -> None:
    from app.services.ingestion.ingestion_service import IngestionService
    from app.models.schemas import IngestRequest

    supported_extensions = {".pdf", ".txt", ".md"}
    files = [f for f in directory.rglob("*") if f.suffix.lower() in supported_extensions]

    print(f"Documents trouvés : {len(files)} dans {directory}")

    if dry_run:
        for f in files:
            print(f"  [DRY-RUN] {f.name}")
        return

    service = IngestionService()
    success, failed = 0, 0

    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file_path.name}...", end=" ")
        try:
            content = file_path.read_bytes()
            request = IngestRequest(
                doc_type=doc_type,
                institution=institution,
                jurisdiction=jurisdiction,
                force_ocr=force_ocr,
            )
            response = await service.ingest_document(
                file_content=content,
                filename=file_path.name,
                request=request,
            )
            print(f"OK — {response.chunks_created} chunks, lang={response.language_detected}")
            success += 1
        except Exception as exc:
            print(f"ERREUR : {exc}")
            failed += 1

    print(f"\nRésultat : {success} ingérés, {failed} échoués sur {len(files)} fichiers.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GOV-AI 2.0 — Ingestion en masse")
    parser.add_argument("--dir", required=True, help="Répertoire contenant les documents")
    parser.add_argument("--doc-type", default="autre", help="Type documentaire (loi, decret, etc.)")
    parser.add_argument("--institution", default="", help="Institution émettrice")
    parser.add_argument("--jurisdiction", default="national", help="Juridiction")
    parser.add_argument("--force-ocr", action="store_true", help="Forcer l'OCR")
    parser.add_argument("--dry-run", action="store_true", help="Lister sans ingérer")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    directory = Path(args.dir)

    if not directory.exists():
        print(f"Répertoire introuvable : {directory}")
        sys.exit(1)

    asyncio.run(ingest_directory(
        directory=directory,
        doc_type=args.doc_type,
        institution=args.institution,
        jurisdiction=args.jurisdiction,
        force_ocr=args.force_ocr,
        dry_run=args.dry_run,
    ))
