"""
GOV-AI 2.0 — Stratégie de chunking documentaire.
Pipeline d'ingestion (Éq. 4.1 du mémoire) :
  d_brut → f_extract → f_clean → f_chunk → {c_1, ..., c_m} → f_embed

Stratégie :
  - STRUCTURAL : un chunk = un article/section (documents à structure formelle)
  - FIXED_SIZE : taille fixe avec overlap (documents non structurés)
  - HYBRID : structural d'abord, fixed_size si l'article est trop long
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.core.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.models.domain import ChunkStrategy
from app.utils.text_utils import count_tokens_approx


@dataclass
class ChunkResult:
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    chunk_index: int = 0
    page: Optional[int] = None
    strategy: str = "fixed_size"
    article_ref: Optional[str] = None
    token_count: int = 0

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = count_tokens_approx(self.content)


# ── Patterns structurels pour les textes législatifs camerounais ──────────────
# Les PDF juridiques camerounais présentent les en-têtes sous des formes variées :
#   « Article 103 : », « ARTICLE 103.- », « Art. 103 », « Section 12 », « Article 5 bis »,
#   « SECTION 25-1 : » (disposition insérée — à ne surtout pas confondre avec « Section 25 »)
# Le libellé est capturé en groupe 1, le numéro (suffixe latin inclus) en groupe 2.
_ARTICLE_KEYWORDS_FR = r"Articles?|Arts?\.?|Alinéas?|Sections?|Chapitres?|Titres?|Parties?|Annexes?|Livres?"
_ARTICLE_KEYWORDS_EN = r"Articles?|Arts?\.?|Sections?|Chapters?|Parts?|Titles?|Schedules?|Books?"

_STRUCTURAL_PATTERNS = [
    re.compile(
        rf"^[ \t]*({_ARTICLE_KEYWORDS_FR})\s*(\d+(?:\s*[-–—]\s*\d+)*(?:\s*(?:bis|ter|quater))?)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        rf"^[ \t]*({_ARTICLE_KEYWORDS_EN})\s*(\d+(?:\s*[-–—]\s*\d+)*(?:\s*(?:bis|ter|quater))?)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
]

# Libellés normalisés : « ART. » et « articles » convergent vers « Article ».
_LABEL_CANONICAL = {
    "article": "Article", "articles": "Article",
    "art": "Article", "art.": "Article", "arts": "Article", "arts.": "Article",
    "alinéa": "Alinéa", "alinéas": "Alinéa",
    "section": "Section", "sections": "Section",
    "chapitre": "Chapitre", "chapitres": "Chapitre",
    "chapter": "Chapter", "chapters": "Chapter",
    "titre": "Titre", "titres": "Titre",
    "title": "Title", "titles": "Title",
    "partie": "Partie", "parties": "Partie",
    "part": "Part", "parts": "Part",
    "annexe": "Annexe", "annexes": "Annexe",
    "schedule": "Schedule", "schedules": "Schedule",
    "livre": "Livre", "livres": "Livre",
    "book": "Book", "books": "Book",
}

# Un en-tête orphelin est un segment qui ne porte QUE la référence d'article :
# « Article 102 : », « Article 104 : (1) ». Le corps se trouve sur la page suivante,
# le découpage page par page les ayant séparés.
_HEADER_ONLY_MAX_CHARS = 48
_HEADER_ONLY_PATTERN = re.compile(
    rf"^[ \t]*(?:{_ARTICLE_KEYWORDS_FR}|{_ARTICLE_KEYWORDS_EN})\s*\d+(?:\s*[-–—]\s*\d+)*(?:\s*(?:bis|ter|quater))?"
    r"[\s:.\-—–]*(?:\(\s*\d+\s*\))?[\s:.\-—–]*$",
    re.IGNORECASE,
)


def _find_structural_splits(text: str) -> list[int]:
    """Trouve les positions des séparateurs structurels (numéros d'articles, etc.)."""
    positions = []
    for pattern in _STRUCTURAL_PATTERNS:
        for match in pattern.finditer(text):
            if match.start() > 0:
                positions.append(match.start())
    return sorted(set(positions))


def chunk_structural(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
    page: Optional[int] = None,
) -> list[ChunkResult]:
    """
    Découpe le texte selon la structure législative (articles, sections).
    Chaque article = un chunk. Si trop long → subdivision fixed_size.
    """
    splits = _find_structural_splits(text)
    if not splits:
        return chunk_fixed_size(text, max_tokens, chunk_overlap_tokens, page)

    # Découper le texte aux positions structurelles
    segments: list[str] = []
    prev = 0
    for pos in splits:
        segment = text[prev:pos].strip()
        if segment:
            segments.append(segment)
        prev = pos
    # Dernier segment
    last_segment = text[prev:].strip()
    if last_segment:
        segments.append(last_segment)

    chunks: list[ChunkResult] = []
    idx = 0
    for segment in segments:
        if not segment.strip():
            continue
        token_count = count_tokens_approx(segment)
        if token_count <= max_tokens:
            article_ref = _extract_article_ref(segment)
            chunks.append(ChunkResult(
                content=segment,
                chunk_index=idx,
                page=page,
                strategy=ChunkStrategy.STRUCTURAL,
                article_ref=article_ref,
                token_count=token_count,
            ))
            idx += 1
        else:
            # Segment trop long → fixed_size
            sub_chunks = chunk_fixed_size(
                segment, max_tokens, chunk_overlap_tokens, page
            )
            for sub in sub_chunks:
                sub.chunk_index = idx
                sub.strategy = ChunkStrategy.HYBRID
                idx += 1
            chunks.extend(sub_chunks)
    return chunks


def chunk_fixed_size(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_SIZE,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
    page: Optional[int] = None,
) -> list[ChunkResult]:
    """
    Découpe le texte en chunks de taille fixe avec overlap.
    Taille de référence : 512 tokens, overlap : 64 tokens (§4.2.2 du mémoire).
    """
    # Estimer les caractères correspondants
    chars_per_token = 4.0
    max_chars = int(max_tokens * chars_per_token)
    overlap_chars = int(overlap_tokens * chars_per_token)

    if len(text) <= max_chars:
        return [ChunkResult(
            content=text,
            chunk_index=0,
            page=page,
            strategy=ChunkStrategy.FIXED_SIZE,
            token_count=count_tokens_approx(text),
        )]

    chunks: list[ChunkResult] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Couper à un espace/newline pour ne pas couper un mot
        if end < len(text):
            for boundary in reversed(range(start + overlap_chars, end)):
                if text[boundary] in " \n":
                    end = boundary
                    break
        content = text[start:end].strip()
        if content:
            chunks.append(ChunkResult(
                content=content,
                chunk_index=idx,
                page=page,
                strategy=ChunkStrategy.FIXED_SIZE,
                token_count=count_tokens_approx(content),
            ))
            idx += 1
        if end >= len(text):
            break
        start = end - overlap_chars  # Overlap
    return chunks


def chunk_hybrid(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_SIZE,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
    page: Optional[int] = None,
) -> list[ChunkResult]:
    """
    Stratégie hybride : structural si la structure est détectée, sinon fixed_size.
    """
    splits = _find_structural_splits(text)
    if splits:
        return chunk_structural(text, max_tokens, overlap_tokens, page)
    return chunk_fixed_size(text, max_tokens, overlap_tokens, page)


def _extract_article_ref(text: str) -> Optional[str]:
    """
    Extrait la référence d'article en tête de segment (ex: 'Article 3', 'Section 12').

    Le libellé est normalisé : « ART. 3 », « art 3 », « Articles 3 » donnent tous
    « Article 3 ». Cette chaîne est la référence citable du chunk : elle est
    persistée puis réinjectée dans le prompt, et le modèle n'a pas le droit d'en
    produire d'autre.
    """
    stripped = text.strip()
    for pattern in _STRUCTURAL_PATTERNS:
        match = pattern.match(stripped)
        if match:
            label = _LABEL_CANONICAL.get(match.group(1).lower(), match.group(1).title())
            number = re.sub(r"\s*[-–—]\s*", "-", match.group(2).strip())
            number = re.sub(r"\s+", " ", number)
            return f"{label} {number}"
    return None


def _is_header_only(text: str) -> bool:
    """
    Vrai si le segment ne contient que l'en-tête d'un article, sans corps.

    Cas produit par le découpage page par page : « Article 102 : » termine une
    page et son contenu commence à la page suivante.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > _HEADER_ONLY_MAX_CHARS:
        return False
    collapsed = re.sub(r"\s+", " ", stripped)
    return bool(_HEADER_ONLY_PATTERN.match(collapsed))


def consolidate_article_chunks(
    chunks: list[ChunkResult],
    language: str = "fr",
) -> list[ChunkResult]:
    """
    Recolle les articles coupés par une frontière de page, puis propage la
    référence d'article aux chunks de continuation.

    Deux défauts corrigés, tous deux dus au chunking page par page :

    1. En-tête orphelin — « Article 102 : » seul en fin de page produit un chunk
       sans contenu (bruit à l'indexation) et laisse son corps sans numéro
       d'article (donc non citable). L'en-tête est fusionné avec le chunk suivant.

    2. Continuation anonyme — un corps d'article qui déborde sur la page suivante
       n'ouvre par aucun en-tête. Il hérite de la référence de l'article en cours,
       tant qu'aucun nouvel en-tête n'apparaît. Cette référence héritée est marquée
       « (suite) » : elle est déduite de l'ordre du document, pas lue dans le texte
       du chunk, et la citation présentée à l'utilisateur doit le refléter.

    La numérotation `chunk_index` est recalculée sur la séquence finale.
    """
    if not chunks:
        return []

    # ── 1. Fusion des en-têtes orphelins avec le chunk suivant ────────────────
    merged: list[ChunkResult] = []
    pending_header: Optional[ChunkResult] = None

    for chunk in chunks:
        if pending_header is not None:
            chunk.content = f"{pending_header.content.strip()}\n{chunk.content.lstrip()}"
            # L'article commence à la page de son en-tête.
            chunk.page = pending_header.page
            chunk.article_ref = pending_header.article_ref or _extract_article_ref(chunk.content)
            chunk.token_count = count_tokens_approx(chunk.content)
            pending_header = None

        if _is_header_only(chunk.content):
            if chunk.article_ref is None:
                chunk.article_ref = _extract_article_ref(chunk.content)
            pending_header = chunk
            continue

        merged.append(chunk)

    # En-tête en toute fin de document : aucun corps à lui rattacher, on le garde tel quel.
    if pending_header is not None:
        merged.append(pending_header)

    # ── 2. Propagation de la référence aux chunks de continuation ─────────────
    continuation_suffix = " (suite)" if language.lower().startswith("fr") else " (cont.)"
    current_ref: Optional[str] = None

    for index, chunk in enumerate(merged):
        own_ref = _extract_article_ref(chunk.content) or chunk.article_ref
        if own_ref and not own_ref.endswith((continuation_suffix, " (suite)", " (cont.)")):
            current_ref = own_ref
            chunk.article_ref = own_ref
        elif current_ref:
            chunk.article_ref = f"{current_ref}{continuation_suffix}"
        chunk.chunk_index = index

    return merged


def chunk_document(
    text: str,
    strategy: str = "hybrid",
    max_tokens: int = DEFAULT_CHUNK_SIZE,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
    page: Optional[int] = None,
) -> list[ChunkResult]:
    """Point d'entrée principal du chunker."""
    if not text or not text.strip():
        return []

    strategy_lower = strategy.lower()
    if strategy_lower == ChunkStrategy.STRUCTURAL:
        return chunk_structural(text, max_tokens, overlap_tokens, page)
    elif strategy_lower == ChunkStrategy.FIXED_SIZE:
        return chunk_fixed_size(text, max_tokens, overlap_tokens, page)
    else:
        return chunk_hybrid(text, max_tokens, overlap_tokens, page)
