"""
GOV-AI 2.0 — Parcours structurel du corpus (requêtes agrégatives).

Une question comme « fais-moi un résumé du Code pénal » n'est pas soluble par le
retrieval top-k : cinq fragments sémantiquement proches ne décrivent pas un code
de 670 articles, et aucune consigne de prompt ne compense ce déficit de matière.

Ce module fournit l'alternative : au lieu de chercher les passages les plus
proches de la question, il reconstitue l'ossature réelle du document — titres,
chapitres, plage d'articles, volume — puis prélève un échantillon d'extraits
réparti sur toute sa longueur. Le modèle décrit alors ce que le document contient
vraiment, et l'étendue couverte est annoncée explicitement plutôt que suggérée.

Cette ossature n'existe que depuis la persistance de `article_ref` à l'ingestion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.core.logging import get_logger
from app.models.schemas import RetrievedChunk

logger = get_logger(__name__)

# En-têtes de structure supra-article, tolérants au bruit OCR :
# « CHAPITRE Il » (II lu comme Il), « CHAPTER || », « TITRE VI ».
_HEADING_PATTERN = re.compile(
    r"^\s*(LIVRE|TITRE|CHAPITRE|PARTIE|BOOK|PART|CHAPTER|TITLE)\s+"
    r"[IVXLCDMivxlcdm0-9|lI]{1,8}\b",
    re.MULTILINE,
)

# Intitulé réduit à son étiquette et son numéro : l'OCR n'a pas produit de titre
# lisible. Doit être signalé comme tel, jamais laissé nu — un modèle comble le
# vide par un titre inventé et vraisemblable.
_LABEL_ONLY_PATTERN = re.compile(
    r"^(LIVRE|TITRE|CHAPITRE|PARTIE|BOOK|PART|CHAPTER|TITLE)\s+[IVXLCDM0-9]+$",
    re.IGNORECASE,
)

# Vocabulaire des tampons officiels apposés sur les scans. Leur présence dans un
# intitulé signale que l'OCR a capté le tampon, pas la structure du texte.
_STAMP_WORDS = {
    "presidence", "republique", "secretariat", "certifiee", "certifie",
    "conforme", "dela", "ampliation", "cabinet", "presidency",
}

# Mots outils légitimement courts : ils ne signalent pas une bribe d'OCR.
_SHORT_WORDS = {
    "de", "du", "des", "la", "le", "les", "et", "en", "au", "aux", "un", "une",
    "of", "the", "and", "in", "on", "to", "for", "by", "or", "no", "i", "ii",
    "iii", "iv", "v", "vi", "vii", "ix", "x",
}

# Un mot du nom de document trop court ou trop courant ne désigne rien.
_STOPWORDS = {
    "de", "du", "des", "la", "le", "les", "un", "une", "et", "en", "au", "aux",
    "loi", "law", "the", "of", "and", "version", "no", "n",
}


@dataclass
class DocumentOutline:
    """Ossature d'un document : ce qu'il contient, et dans quel ordre."""
    doc_id: str
    source: str
    language: str
    doc_type: str
    total_chunks: int
    article_count: int
    first_article: Optional[str] = None
    last_article: Optional[str] = None
    page_min: Optional[int] = None
    page_max: Optional[int] = None
    headings: list[tuple[int, str]] = field(default_factory=list)  # (page, intitulé)

    def render(self, language: str = "fr") -> str:
        """
        Rend l'ossature sous forme textuelle, pour injection dans le prompt.

        Un intitulé réduit à son étiquette (« CHAPTER VII ») parce que l'OCR n'a
        pas produit de titre lisible est annoté comme tel. Laisser l'étiquette
        nue crée un vide que le modèle comble par un titre inventé et
        vraisemblable — défaut observé, plus trompeur que l'absence d'intitulé.
        """
        illegible = (
            "[heading illegible in the scan]" if language.startswith("en")
            else "[intitulé illisible dans le scan]"
        )

        def _label(title: str) -> str:
            return f"{title} — {illegible}" if _LABEL_ONLY_PATTERN.match(title) else title

        if language.startswith("en"):
            lines = [
                f"DOCUMENT: {self.source}",
                f"Type: {self.doc_type or 'unspecified'} | Language: {self.language}",
                f"Volume: {self.article_count} distinct provisions, "
                f"{self.total_chunks} indexed passages, pages {self.page_min}-{self.page_max}",
            ]
            if self.first_article and self.last_article:
                lines.append(f"Provision range: {self.first_article} … {self.last_article}")
            if self.headings:
                lines.append(
                    f"Structure as it appears in the document — CLOSED list of "
                    f"{len(self.headings)} entries, nothing else exists:"
                )
                lines += [
                    f"  [H{i}] p. {page} — {_label(title)}"
                    for i, (page, title) in enumerate(self.headings, start=1)
                ]
        else:
            lines = [
                f"DOCUMENT : {self.source}",
                f"Type : {self.doc_type or 'non précisé'} | Langue : {self.language}",
                f"Volume : {self.article_count} dispositions distinctes, "
                f"{self.total_chunks} passages indexés, pages {self.page_min} à {self.page_max}",
            ]
            if self.first_article and self.last_article:
                lines.append(f"Plage des dispositions : {self.first_article} … {self.last_article}")
            if self.headings:
                lines.append(
                    f"Structure telle qu'elle figure dans le document — liste CLOSE de "
                    f"{len(self.headings)} entrées, il n'en existe pas d'autres :"
                )
                lines += [
                    f"  [H{i}] p. {page} — {_label(title)}"
                    for i, (page, title) in enumerate(self.headings, start=1)
                ]
        return "\n".join(lines)


def _clean_heading(raw: str) -> Optional[str]:
    """
    Nettoie un intitulé de structure issu d'un document OCRisé.

    Les scans portent des tampons et des mentions manuscrites qui débordent dans
    le texte : « CHAPTER Vil SAZLOENCE DE LA REPUBLIQUE ya ». Transmettre cela au
    modèle l'invite à présenter du charabia comme le plan officiel du code.

    L'étiquette et son numéro sont conservés — ils sont fiables. La partie
    descriptive n'est gardée que si elle est lisible ; sinon l'intitulé est réduit
    à « CHAPTER IV », ce qui reste exact quoique moins informatif.
    """
    match = _HEADING_PATTERN.search(raw)
    if not match:
        return None

    label = match.group(1).upper()
    remainder = raw[match.end(1):].strip()

    # Numéro : les chiffres romains sont massacrés par l'OCR (I lu l, | ou 1).
    numeral_match = re.match(r"[IVXLCDMivxlcdm0-9|lI!]{1,8}", remainder)
    numeral = ""
    if numeral_match:
        numeral = (
            numeral_match.group(0)
            .replace("|", "I").replace("l", "I").replace("!", "I")
            .upper()
        )
        remainder = remainder[numeral_match.end():]

    # Partie descriptive : uniquement lettres, espaces et ponctuation légère.
    description = re.sub(r"[^\w\s'’\-]", " ", remainder, flags=re.UNICODE)
    description = re.sub(r"\s+", " ", description).strip()

    words = [w for w in description.split() if w]
    if words:
        # Règle 1 — tampon officiel : les scans portent « PRÉSIDENCE DE LA
        # RÉPUBLIQUE », « CERTIFIÉE CONFORME » etc., qui n'ont rien d'un intitulé.
        if any(_normalize(w) in _STAMP_WORDS for w in words):
            words = []
        else:
            # Règle 2 — bribes OCR : couper à la première miette (« cea », « ya »,
            # « uaue ») qui n'est pas un mot outil légitime.
            kept: list[str] = []
            for word in words:
                low = _normalize(word)
                if len(low) <= 3 and low not in _SHORT_WORDS:
                    break
                kept.append(word)
            words = kept

        plausible = sum(1 for w in words if w.isalpha() and len(w) > 1)
        if not words or plausible < 2:
            description = ""
        else:
            description = " ".join(words[:10])

    title = " ".join(part for part in (label, numeral, description) if part).strip()
    return title[:90] if title else None


def _normalize(text: str) -> str:
    """Minuscules sans accents ni ponctuation, pour l'appariement de noms."""
    import unicodedata

    stripped = "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", " ", stripped).strip()


def _significant_words(text: str) -> set[str]:
    return {
        w for w in _normalize(text).split()
        if len(w) > 2 and w not in _STOPWORDS
    }


async def list_documents() -> list[dict]:
    """Tous les documents du corpus avec leur volumétrie."""
    import sqlalchemy as sa

    from app.models.db_models import Chunk, Document
    from app.storage.postgres_client import get_db_session

    async with get_db_session() as db:
        rows = (await db.execute(
            sa.select(
                Document.id, Document.source, Document.language, Document.doc_type,
                sa.func.count(Chunk.id).label("chunks"),
            )
            .join(Chunk, Chunk.doc_id == Document.id, isouter=True)
            .group_by(Document.id, Document.source, Document.language, Document.doc_type)
        )).all()

    return [
        {
            "doc_id": r.id,
            "source": r.source,
            "language": r.language,
            "doc_type": r.doc_type or "",
            "chunks": r.chunks,
        }
        for r in rows
    ]


def resolve_target_documents(query: str, documents: list[dict]) -> list[dict]:
    """
    Détermine sur quel(s) document(s) porte une requête agrégative.

    L'appariement se fait sur les mots significatifs partagés entre la question et
    le nom du document. Sans recouvrement, on ne devine pas : tous les documents
    sont retournés, et la réponse couvrira le corpus entier.
    """
    query_words = _significant_words(query)
    if not query_words:
        return documents

    scored: list[tuple[int, dict]] = []
    for doc in documents:
        overlap = len(query_words & _significant_words(doc["source"]))
        if overlap:
            scored.append((overlap, doc))

    if not scored:
        return documents

    best = max(score for score, _ in scored)
    return [doc for score, doc in scored if score == best]


async def build_outline(doc_id: str, max_headings: int = 40) -> Optional[DocumentOutline]:
    """Reconstitue l'ossature d'un document depuis PostgreSQL."""
    import sqlalchemy as sa

    from app.models.db_models import Chunk, Document
    from app.storage.postgres_client import get_db_session

    async with get_db_session() as db:
        doc = (await db.execute(
            sa.select(Document).where(Document.id == doc_id)
        )).scalar_one_or_none()
        if doc is None:
            return None

        stats = (await db.execute(
            sa.select(
                sa.func.count(Chunk.id),
                sa.func.count(sa.distinct(Chunk.article_ref)),
                sa.func.min(Chunk.page),
                sa.func.max(Chunk.page),
            ).where(Chunk.doc_id == doc_id)
        )).one()

        # Première et dernière disposition, dans l'ordre du document
        bounds = (await db.execute(
            sa.select(Chunk.article_ref)
            .where(Chunk.doc_id == doc_id, Chunk.article_ref.isnot(None))
            .order_by(Chunk.chunk_index)
        )).scalars().all()

        heading_rows = (await db.execute(
            sa.select(Chunk.page, Chunk.content)
            .where(Chunk.doc_id == doc_id)
            .order_by(Chunk.chunk_index)
        )).all()

    headings: list[tuple[int, str]] = []
    seen: set[str] = set()
    for page, content in heading_rows:
        match = _HEADING_PATTERN.search(content or "")
        if not match:
            continue
        # L'étiquette et son titre sont sur des lignes distinctes, souvent séparées
        # par une ligne vide (« TITRE VI » / «  » / « DES AMENDES FORFAITAIRES »).
        # Compter les lignes physiques ferait perdre le titre : on prend les deux
        # premières lignes NON VIDES.
        lines = [
            line.strip()
            for line in content[match.start():].split("\n")
            if line.strip()
        ]
        raw_title = " ".join(lines[:2])
        title = _clean_heading(re.sub(r"\s+", " ", raw_title))
        if not title:
            continue
        key = _normalize(title)
        if not key or key in seen:
            continue
        seen.add(key)
        headings.append((page or 0, title))
        if len(headings) >= max_headings:
            break

    return DocumentOutline(
        doc_id=doc.id,
        source=doc.source,
        language=doc.language,
        doc_type=doc.doc_type or "",
        total_chunks=stats[0] or 0,
        article_count=stats[1] or 0,
        first_article=bounds[0] if bounds else None,
        last_article=bounds[-1] if bounds else None,
        page_min=stats[2],
        page_max=stats[3],
        headings=headings,
    )


async def sample_chunks(doc_id: str, sample_size: int = 12) -> list[RetrievedChunk]:
    """
    Prélève des extraits répartis sur toute la longueur du document.

    Le retrieval top-k concentre les extraits autour d'un point du document ;
    ici on échantillonne à intervalle régulier pour que le modèle voie du contenu
    du début à la fin, et non une seule matière.
    """
    import sqlalchemy as sa

    from app.models.db_models import Chunk, Document
    from app.models.domain import Language
    from app.storage.postgres_client import get_db_session

    async with get_db_session() as db:
        doc = (await db.execute(
            sa.select(Document).where(Document.id == doc_id)
        )).scalar_one_or_none()
        if doc is None:
            return []

        rows = (await db.execute(
            sa.select(Chunk)
            .where(Chunk.doc_id == doc_id)
            .order_by(Chunk.chunk_index)
        )).scalars().all()

    if not rows:
        return []

    if len(rows) <= sample_size:
        selected = list(rows)
    else:
        step = len(rows) / sample_size
        selected = [rows[min(int(i * step), len(rows) - 1)] for i in range(sample_size)]

    try:
        language = Language(doc.language)
    except ValueError:
        language = Language.UNKNOWN

    return [
        RetrievedChunk(
            chunk_id=c.id,
            doc_id=doc.id,
            content=c.content,
            source=doc.source,
            language=language,
            page=c.page,
            chunk_index=c.chunk_index,
            final_score=0.0,
            metadata={
                "doc_type": doc.doc_type or "",
                "institution": doc.institution or "",
                "jurisdiction": doc.jurisdiction or "",
                "article_ref": c.article_ref or "",
            },
        )
        for c in selected
    ]


async def build_survey(
    query: str, sample_size: int = 12
) -> tuple[str, list[RetrievedChunk], dict[str, set[int]]]:
    """
    Prépare une requête agrégative : ossature textuelle + extraits représentatifs.

    Returns:
        (ossature rendue, extraits, pages attestées par l'index)

    L'ossature va dans le prompt comme description factuelle du corpus, les
    extraits comme sources citables. Le troisième élément recense, par document,
    les pages que l'index atteste : un modèle cite spontanément l'ossature sous
    la forme [Source: doc, p. 3], référence exacte mais non adossée à un extrait.
    Sans cette liste, la vérification la signalerait comme une hallucination.
    """
    documents = await list_documents()
    if not documents:
        return "", [], {}

    targets = resolve_target_documents(query, documents)
    per_doc = max(3, sample_size // max(len(targets), 1))

    outlines: list[str] = []
    chunks: list[RetrievedChunk] = []
    index_pages: dict[str, set[int]] = {}
    for doc in targets:
        outline = await build_outline(doc["doc_id"])
        if outline is None:
            continue
        outlines.append(outline.render(outline.language))
        chunks.extend(await sample_chunks(doc["doc_id"], per_doc))
        index_pages[outline.source] = {page for page, _ in outline.headings if page}

    logger.info(
        "structural_survey_built",
        documents=[d["source"] for d in targets],
        chunks_sampled=len(chunks),
    )
    return "\n\n".join(outlines), chunks, index_pages
