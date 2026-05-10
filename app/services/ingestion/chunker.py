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
_STRUCTURAL_PATTERNS = [
    re.compile(
        r"^(Article|Alinéa|Section|Chapitre|Titre|Partie|Annexe)\s+\d+",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^(Article|Section|Chapter|Part|Title|Schedule)\s+\d+",
        re.IGNORECASE | re.MULTILINE,
    ),
]


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
    """Extrait la référence d'article (ex: 'Article 3', 'Section 12')."""
    match = re.match(
        r"^(Article|Section|Chapter|Alinéa|Chapitre|Titre)\s+(\d+\w*)",
        text.strip(),
        re.IGNORECASE,
    )
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None


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
