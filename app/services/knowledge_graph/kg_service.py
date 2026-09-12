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
    institutions: list[dict[str, Any]] = field(default_factory=list)
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
        self._concepts_present: Optional[bool] = None

    async def _has_concepts(self) -> bool:
        """
        Le label `Concept` est-il peuplé ? Sondé une fois, puis mémorisé.

        La réponse ne change qu'au peuplement d'une terminologie juridique,
        c'est-à-dire au redémarrage du service qui l'écrira.
        """
        if self._concepts_present is not None:
            return self._concepts_present
        try:
            from app.storage.neo4j_client import run_query
            rows = await run_query("MATCH (c:Concept) RETURN count(c) AS n LIMIT 1")
            self._concepts_present = bool(rows and rows[0].get("n"))
        except Exception:
            self._concepts_present = False
        return self._concepts_present

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
        article_ids: Optional[list[str]] = None,
    ) -> KGContext:
        """
        Enrichit le contexte à partir des entités extraites.

        Args:
            entities: Entités détectées (ex. "article 314", "décret n°87-1872")
            query: Requête originale
            language: Langue (fr/en)
            article_ids: identifiants des articles retrouvés par le retrieval,
                sous la forme « {doc_id}:{numéro} ». Ils servent à la traversée
                inverse : une question comme « quelle autorité peut décerner un
                mandat de détention provisoire » ne nomme aucune institution,
                mais les articles retrouvés, eux, en nomment une.

        Returns:
            KGContext avec articles liés, concepts, institutions, hiérarchie
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

        # Institutions nommées dans la question, et articles qui les citent.
        # C'est la voie propre pour « quelle autorité peut… » : la compétence
        # est portée par une arête, non par une proximité vectorielle.
        ctx.institutions = await self._fetch_institutions(query)

        # Traversée inverse. Une institution déjà nommée dans la question n'est
        # pas redite : l'intention explicite prime sur la déduction.
        if article_ids:
            deja = {i.get("institution_id") for i in ctx.institutions}
            deduites = await self._fetch_institutions_of_articles(article_ids)
            ctx.institutions.extend(
                i for i in deduites if i.get("institution_id") not in deja
            )

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
            institutions=len(ctx.institutions),
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

    async def _fetch_institutions(self, query: str) -> list[dict[str, Any]]:
        """
        Institutions nommées dans la question, avec les articles qui les citent.

        Le rattachement part du vocabulaire curé plutôt que d'une recherche
        plein texte : « le procureur » et « the State Counsel » désignent le
        même organe, et une correspondance approximative sur le libellé
        confondrait le Procureur de la République et le Procureur Général.
        """
        from app.services.knowledge_graph.institutions import find_institutions

        ids = find_institutions(query)
        if not ids:
            return []
        try:
            from app.storage.neo4j_client import run_query
            results = await run_query(
                """
                UNWIND $ids AS iid
                MATCH (i:Institution {id: iid})
                OPTIONAL MATCH (t:TexteNormatif)-[:A_ARTICLE]->(a:Article)-[:MENTIONNE]->(i)
                WITH i, [x IN collect(DISTINCT {number: a.number, doc: t.title})
                         WHERE x.number IS NOT NULL] AS cites
                OPTIONAL MATCH (i)-[:EMETRICE_DE]->(emis:TexteNormatif)
                RETURN i.id AS institution_id,
                       i.label AS label,
                       i.category AS category,
                       cites AS articles,
                       size(cites) AS article_count,
                       collect(DISTINCT emis.title) AS textes_emis,
                       'corpus' AS scope
                """,
                {"ids": ids[:3]},
            )
            # Le tri se fait ici et non en Cypher : les numéros sont des chaînes
            # (« 5 bis », « 25-1 »), et toInteger y renvoie null.
            for row in results:
                row["articles"] = sorted(
                    row.get("articles") or [], key=self._article_sort_key
                )[:8]
            return results
        except Exception as exc:
            logger.debug("kg_fetch_institutions_failed", error=str(exc))
            return []

    async def _fetch_institutions_of_articles(
        self, article_ids: list[str]
    ) -> list[dict[str, Any]]:
        """
        Traversée inverse : quelles autorités les articles retrouvés nomment-ils ?

        C'est la voie qui répond à « quelle autorité peut décerner un mandat de
        dépôt ». La question ne nomme aucune institution — le rapprochement par
        libellé est donc inopérant. Mais l'article que le retrieval remonte, lui,
        nomme le juge d'instruction, et l'arête le dit sans que le modèle ait à
        le déduire d'un paragraphe.

        Le décompte porte ici sur les seuls articles retrouvés, non sur le
        corpus entier : annoncer « cité dans 89 articles » à propos de deux
        passages induirait en erreur.
        """
        if not article_ids:
            return []
        try:
            from app.storage.neo4j_client import run_query
            results = await run_query(
                """
                UNWIND $ids AS aid
                MATCH (t:TexteNormatif)-[:A_ARTICLE]->(a:Article {id: aid})
                      -[:MENTIONNE]->(i:Institution)
                WITH i, collect(DISTINCT {number: a.number, doc: t.title}) AS cites
                RETURN i.id AS institution_id,
                       i.label AS label,
                       i.category AS category,
                       cites AS articles,
                       size(cites) AS article_count,
                       [] AS textes_emis,
                       'passages' AS scope
                ORDER BY article_count DESC
                LIMIT 4
                """,
                {"ids": article_ids[:20]},
            )
            for row in results:
                row["articles"] = sorted(
                    row.get("articles") or [], key=self._article_sort_key
                )[:8]
            return results
        except Exception as exc:
            logger.debug("kg_fetch_institutions_of_articles_failed", error=str(exc))
            return []

    @staticmethod
    def _article_sort_key(entry: dict[str, Any]) -> tuple[int, str]:
        number = str(entry.get("number") or "")
        digits = "".join(c for c in number if c.isdigit())
        return (int(digits) if digits else 10**6, number)

    async def _fetch_related_concepts(self, query: str, language: str) -> list[dict[str, Any]]:
        """
        Récupère les concepts juridiques liés à la requête.

        Aucun nœud `Concept` n'est écrit à ce jour : le label est déclaré au
        schéma, contraint et indexé, mais le peuplement de la terminologie
        juridique reste à faire. La sonde ci-dessous évite d'interroger un
        label vide à chaque question — ce qui n'avait d'autre effet que de
        faire journaliser à Neo4j un avertissement sur `DEFINI_DANS`,
        relation qui n'existe pas davantage.
        """
        if not await self._has_concepts():
            return []
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

    def kg_context_to_evidence(self, ctx: KGContext) -> list[Any]:
        """
        Sérialise le contexte du graphe pour l'affichage, non pour le prompt.

        Pendant du `kg_context_to_prompt_str` : celui-ci écrit un bloc de texte
        que le modèle lit, celle-ci produit des objets que l'interface montre.
        Les deux doivent rester cohérents — ce que le graphe a dit au modèle est
        exactement ce que l'utilisateur doit pouvoir inspecter.
        """
        from app.models.schemas import GraphArticleRef, GraphEvidence

        evidence: list[GraphEvidence] = []

        for inst in ctx.institutions:
            evidence.append(GraphEvidence(
                kind="institution",
                label=inst.get("label") or "",
                detail=inst.get("category"),
                scope=inst.get("scope"),
                articles=[
                    GraphArticleRef(
                        number=str(a.get("number") or ""),
                        doc_title=str(a.get("doc") or ""),
                    )
                    for a in (inst.get("articles") or [])
                ],
                article_count=int(inst.get("article_count") or 0),
                issued_texts=[t for t in (inst.get("textes_emis") or []) if t],
            ))

        for art in ctx.articles:
            evidence.append(GraphEvidence(
                kind="article",
                label=art.title or f"Article {art.number}",
                detail=art.doc_title,
                # Les renvois sortants sont la relation que le graphe apporte :
                # elle n'apparaît dans aucun extrait pris isolément.
                article_count=len(art.related_articles),
            ))

        return evidence

    def kg_context_to_prompt_str(self, ctx: KGContext, language: str = "fr") -> str:
        """Sérialise le contexte KG en bloc texte pour le prompt LLM."""
        if not ctx.articles and not ctx.concepts and not ctx.institutions:
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

        for inst in ctx.institutions:
            shown = inst.get("articles", [])
            count = inst.get("article_count", 0)
            refs = ", ".join(f"art. {a['number']} ({a['doc']})" for a in shown)
            label = inst.get("label", "")

            # Deux portées, deux phrases : « 66 articles du corpus » et
            # « les passages retrouvés » ne disent pas la même chose, et les
            # confondre ferait annoncer au modèle un décompte qu'il n'a pas.
            if inst.get("scope") == "passages":
                head = (
                    f"\n[INST] {label} — named by the retrieved provisions"
                    if language == "en"
                    else f"\n[INST] {label} — nommée par les dispositions retrouvées"
                )
            else:
                head = (
                    f"\n[INST] {label} — cited in {count} article(s) of the corpus"
                    if language == "en"
                    else f"\n[INST] {label} — cité dans {count} article(s) du corpus"
                )
            line = head + (f" : {refs}" if refs else "")
            if count > len(shown):
                line += " (partial list)" if language == "en" else " (liste partielle)"

            emis = [t for t in inst.get("textes_emis", []) if t]
            if emis:
                mot = "issuer of" if language == "en" else "émetteur de"
                line += f"\n  {mot} : {', '.join(emis)}"
            parts.append(line)

        if ctx.concepts:
            concept_labels = [c.get("label", "") for c in ctx.concepts[:3]]
            parts.append(f"\nConcepts juridiques liés : {', '.join(concept_labels)}")

        return "\n".join(parts)
