"""
GOV-AI 2.0 — Service de graphe de connaissances juridiques (Sprint 2).

Fonctions :
  - Enrichissement des entités extraites par l'orchestrateur
  - Récupération des articles liés (références croisées, abrogations)
  - Contexte normatif hiérarchique (Constitution → Loi → Décret → Arrêté)
  - Détection des conflits civil/common law
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KGArticle:
    """Représente un article extrait du graphe."""
    article_id: str
    number: str
    title: str
    content_preview: str
    doc_title: str
    doc_type: str
    jurisdiction: str  # "civil_law" | "common_law" | "both"
    score: float = 1.0
    related_articles: list[str] = field(default_factory=list)


@dataclass
class KGContext:
    """Contexte juridique enrichi par le graphe."""
    articles: list[KGArticle] = field(default_factory=list)
    concepts: list[dict[str, Any]] = field(default_factory=list)
    hierarchical_chain: list[str] = field(default_factory=list)
    bijuridical_divergences: list[dict[str, Any]] = field(default_factory=list)


class KnowledgeGraphService:
    """
    Interroge le graphe Neo4j pour enrichir le contexte RAG.

    Implémente la couche KG de l'Algo 3 (§4.3.3 du mémoire) :
      - Traversée hiérarchie normative
      - Résolution des références croisées entre articles
      - Détection des divergences bijuridiques (civil law / common law)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._available: Optional[bool] = None

    async def _is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from app.storage.neo4j_client import check_neo4j_connection
            self._available = await check_neo4j_connection()
        except Exception:
            self._available = False
        return self._available

    async def enrich_context(
        self,
        entities: list[str],
        query: str,
        language: str = "fr",
    ) -> KGContext:
        """
        Enrichit le contexte à partir des entités extraites.

        Args:
            entities: Entités détectées (ex. "article 314", "décret n°87-1872")
            query: Requête originale
            language: Langue (fr/en)

        Returns:
            KGContext avec articles liés, concepts, chaîne hiérarchique
        """
        if not await self._is_available():
            logger.debug("kg_service_unavailable_fallback")
            return KGContext()

        ctx = KGContext()

        for entity in entities:
            entity_lower = entity.lower()
            # Articles explicites
            if "article" in entity_lower:
                number = entity_lower.replace("article", "").strip()
                articles = await self._fetch_articles_by_number(number)
                ctx.articles.extend(articles)

            # Textes normatifs (décrets, lois)
            elif any(kw in entity_lower for kw in ["loi", "décret", "arrêté", "code", "law", "decree"]):
                articles = await self._fetch_articles_by_text_ref(entity)
                ctx.articles.extend(articles)

        # Concepts liés à la requête
        ctx.concepts = await self._fetch_related_concepts(query, language)

        # Chaîne hiérarchique normative (si articles trouvés)
        if ctx.articles:
            ctx.hierarchical_chain = await self._build_hierarchy_chain(ctx.articles)

        # Divergences bijuridiques
        ctx.bijuridical_divergences = await self._detect_bijuridical_divergences(ctx.articles)

        logger.info(
            "kg_context_built",
            articles=len(ctx.articles),
            concepts=len(ctx.concepts),
            divergences=len(ctx.bijuridical_divergences),
        )
        return ctx

    async def _fetch_articles_by_number(self, number: str) -> list[KGArticle]:
        """Récupère les articles par numéro (ex. '314', '312 al.2')."""
        try:
            from app.storage.neo4j_client import run_query
            results = await run_query(
                """
                MATCH (a:Article)-[:A_ARTICLE]-(t:TexteNormatif)
                WHERE a.number CONTAINS $number
                OPTIONAL MATCH (a)-[:REFERENCE]->(ref:Article)
                RETURN a.id AS article_id,
                       a.number AS number,
                       a.title AS title,
                       a.content_preview AS content_preview,
                       t.title AS doc_title,
                       t.doc_type AS doc_type,
                       t.jurisdiction AS jurisdiction,
                       collect(ref.id) AS related_articles
                LIMIT 5
                """,
                {"number": number},
            )
            return [self._result_to_kg_article(r) for r in results]
        except Exception as exc:
            logger.debug("kg_fetch_articles_failed", error=str(exc))
            return []

    async def _fetch_articles_by_text_ref(self, text_ref: str) -> list[KGArticle]:
        """Récupère les articles d'un texte normatif référencé."""
        try:
            from app.storage.neo4j_client import run_query
            results = await run_query(
                """
                MATCH (t:TexteNormatif)-[:A_ARTICLE]->(a:Article)
                WHERE toLower(t.title) CONTAINS toLower($ref)
                   OR toLower(t.reference) CONTAINS toLower($ref)
                RETURN a.id AS article_id,
                       a.number AS number,
                       a.title AS title,
                       a.content_preview AS content_preview,
                       t.title AS doc_title,
                       t.doc_type AS doc_type,
                       t.jurisdiction AS jurisdiction,
                       [] AS related_articles
                LIMIT 10
                """,
                {"ref": text_ref},
            )
            return [self._result_to_kg_article(r) for r in results]
        except Exception as exc:
            logger.debug("kg_fetch_text_failed", error=str(exc))
            return []

    async def _fetch_related_concepts(self, query: str, language: str) -> list[dict[str, Any]]:
        """Récupère les concepts juridiques liés à la requête."""
        try:
            from app.storage.neo4j_client import run_query
            # Recherche par mots-clés dans les concepts
            words = [w for w in query.lower().split() if len(w) > 4][:5]
            if not words:
                return []
            results = await run_query(
                """
                MATCH (c:Concept)
                WHERE any(word IN $words WHERE toLower(c.label) CONTAINS word)
                OPTIONAL MATCH (c)-[:DEFINI_DANS]->(a:Article)
                RETURN c.id AS concept_id,
                       c.label AS label,
                       c.definition AS definition,
                       c.language AS language,
                       collect(a.number) AS defined_in_articles
                LIMIT 5
                """,
                {"words": words},
            )
            return results
        except Exception as exc:
            logger.debug("kg_fetch_concepts_failed", error=str(exc))
            return []

    async def _build_hierarchy_chain(self, articles: list[KGArticle]) -> list[str]:
        """Construit la chaîne hiérarchique normative pour les articles trouvés."""
        hierarchy_order = [
            "constitution", "loi_organique", "loi", "ordonnance",
            "decret", "arrete", "circulaire", "acte_ohada",
        ]
        doc_types = {a.doc_type.lower() for a in articles if a.doc_type}
        chain = [dt for dt in hierarchy_order if dt in doc_types]
        return chain

    async def _detect_bijuridical_divergences(
        self, articles: list[KGArticle]
    ) -> list[dict[str, Any]]:
        """Détecte les articles avec divergences civil law / common law."""
        divergences = []
        civil_articles = [a for a in articles if a.jurisdiction == "civil_law"]
        common_articles = [a for a in articles if a.jurisdiction == "common_law"]

        if civil_articles and common_articles:
            divergences.append({
                "type": "bijuridical_divergence",
                "civil_law_articles": [a.number for a in civil_articles],
                "common_law_articles": [a.number for a in common_articles],
                "note": "Des dispositions différentes existent selon le système juridique applicable.",
            })
        return divergences

    def _result_to_kg_article(self, r: dict[str, Any]) -> KGArticle:
        return KGArticle(
            article_id=r.get("article_id", ""),
            number=r.get("number", ""),
            title=r.get("title", ""),
            content_preview=r.get("content_preview", ""),
            doc_title=r.get("doc_title", ""),
            doc_type=r.get("doc_type", ""),
            jurisdiction=r.get("jurisdiction", "both"),
            related_articles=r.get("related_articles", []),
        )

    def kg_context_to_prompt_str(self, ctx: KGContext, language: str = "fr") -> str:
        """Sérialise le contexte KG en bloc texte pour le prompt LLM."""
        if not ctx.articles and not ctx.concepts:
            return ""

        parts: list[str] = []
        if language == "en":
            parts.append("=== KNOWLEDGE GRAPH CONTEXT ===")
        else:
            parts.append("=== CONTEXTE DU GRAPHE DE CONNAISSANCES ===")

        for i, art in enumerate(ctx.articles, 1):
            parts.append(
                f"\n[KG-{i}] {art.doc_title}, Art. {art.number}"
                + (f" — {art.title}" if art.title else "")
                + f"\nJuridiction: {art.jurisdiction}"
                + (f"\n{art.content_preview}" if art.content_preview else "")
            )

        if ctx.bijuridical_divergences:
            for div in ctx.bijuridical_divergences:
                parts.append(
                    f"\n⚖️ Divergence bijuridique détectée : "
                    f"droit civil (art. {', '.join(div['civil_law_articles'])}) vs "
                    f"common law (art. {', '.join(div['common_law_articles'])})"
                )

        if ctx.concepts:
            concept_labels = [c.get("label", "") for c in ctx.concepts[:3]]
            parts.append(f"\nConcepts juridiques liés : {', '.join(concept_labels)}")

        return "\n".join(parts)
