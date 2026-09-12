"""
GOV-AI 2.0 — Recherche web restreinte aux sources officielles camerounaises.

Conception dictée par la mesure, non par la théorie :

- L'opérateur `site:` du moteur est ignoré ou renvoie zéro résultat. Le filtrage
  se fait donc **sur les résultats**, jamais sur la requête — seul point de
  contrôle indépendant du comportement du moteur.
- La région `fr-FR` oriente vers la France : « passeport » y ramenait
  `service-public.gouv.fr` et l'ANTS. La région mondiale (`wt-wt`) est la seule
  qui laisse remonter les domaines camerounais.
- **Nommer l'institution compétente est ce qui fait apparaître les sites
  officiels.** Mesuré : « MINFI Cameroun taux TVA » ramène 7 résultats en .cm
  sur 8 ; « Cameroun création entreprise » n'en ramène aucun.

Un résultat hors liste blanche est écarté, jamais présenté comme officiel.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.websearch.official_sources import (
    OfficialSource,
    Priority,
    Routing,
    find_source,
    route_question,
)

logger = get_logger(__name__)

# La région mondiale est la seule qui laisse remonter les domaines .cm.
# « fr-FR » désigne la France et détourne la recherche vers ses administrations.
_REGION = "wt-wt"

# Au-delà, la requête devient trop spécifique et le moteur ne renvoie plus rien.
_MAX_INSTITUTIONS_PER_QUERY = 2

# Le moteur limite le débit et répond par une liste vide plutôt que par une
# erreur. Ces temporisations ont été calibrées sur des rafales de tests qui
# faisaient tomber les résultats à zéro.
_INTER_QUERY_DELAY_S = 1.5
_RETRY_DELAY_S = 2.0
_MAX_ATTEMPTS = 2


@dataclass
class WebResult:
    """Un résultat retenu, rattaché à son institution d'origine."""
    title: str
    url: str
    snippet: str
    source: OfficialSource

    @property
    def domain(self) -> str:
        return self.source.domain

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "domain": self.source.domain,
            "institution": self.source.institution,
            "acronym": self.source.acronym,
            "priority": int(self.source.priority),
            "reachable": self.source.reachable,
        }


@dataclass
class WebSearchOutcome:
    """Ce que la recherche a produit, et ce qu'elle a écarté."""
    results: list[WebResult] = field(default_factory=list)
    #: Pages lues directement sur les sites, contenu intégral et non résumé.
    pages_read: int = 0
    routing: Optional[Routing] = None
    queries: list[str] = field(default_factory=list)
    discarded: int = 0
    error: Optional[str] = None
    elapsed_s: float = 0.0

    @property
    def found(self) -> bool:
        return bool(self.results)


class WebSearchService:
    """Recherche web filtrée par la liste blanche des sources officielles."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def build_queries(self, question: str, routing: Routing) -> list[str]:
        """
        Compose les requêtes en nommant les institutions compétentes.

        Une requête par institution plutôt qu'une requête fourre-tout : le
        moteur renvoie zéro résultat dès que la requête accumule trop de termes.
        """
        queries: list[str] = []
        for source in routing.sources[:_MAX_INSTITUTIONS_PER_QUERY]:
            queries.append(f"{source.query_hint} {question}".strip())
        if not queries:
            queries.append(f"Cameroun administration {question}".strip())
        return queries

    async def search(self, question: str, max_results: Optional[int] = None) -> WebSearchOutcome:
        """
        Route la question, interroge le moteur, ne conserve que l'officiel.

        Returns:
            WebSearchOutcome — résultats retenus, institutions consultées, et
            nombre de résultats écartés parce que hors liste blanche.
        """
        started = time.perf_counter()
        limit = max_results or self.settings.web_search_max_results
        routing = route_question(question)
        queries = self.build_queries(question, routing)

        outcome = WebSearchOutcome(routing=routing, queries=queries)

        # ── Voie principale : lire les sites institutionnels directement ──────
        # Elle ne dépend d'aucun tiers, ne subit aucune limitation de débit, et
        # rend le contenu réel des pages plutôt qu'un extrait de 500 caractères
        # produit par un moteur.
        from app.services.websearch.site_reader import get_site_reader

        try:
            extracts = await get_site_reader().read_institutions(
                routing.sources, question, max_pages_total=limit
            )
        except Exception as exc:
            extracts = []
            logger.warning("site_reading_failed", error=str(exc))

        for extract in extracts:
            outcome.results.append(WebResult(
                title=extract.title,
                url=extract.url,
                snippet=extract.text,
                source=extract.source,
            ))
        outcome.pages_read = len(extracts)

        if len(outcome.results) >= limit:
            outcome.elapsed_s = round(time.perf_counter() - started, 2)
            logger.info(
                "web_search_completed",
                mode="lecture directe",
                institutions=[s.acronym for s in routing.sources],
                pages_read=outcome.pages_read,
                elapsed_s=outcome.elapsed_s,
            )
            return outcome

        # ── Complément : le moteur, si la lecture directe n'a pas suffi ───────

        try:
            raw = await asyncio.to_thread(self._search_sync, queries, limit)
        except ImportError:
            outcome.error = (
                "Le module de recherche web n'est pas installé sur ce serveur "
                "(duckduckgo-search)."
            )
            logger.warning("web_search_unavailable")
            outcome.elapsed_s = round(time.perf_counter() - started, 2)
            return outcome
        except Exception as exc:
            outcome.error = f"Le moteur de recherche n'a pas répondu : {exc}"
            logger.warning("web_search_failed", error=str(exc))
            outcome.elapsed_s = round(time.perf_counter() - started, 2)
            return outcome

        # Une requête mono-institution revient parfois vide : le moteur ne
        # renvoie rien plutôt que des résultats approchants. On réessaie alors
        # en désignant le pays plutôt que l'institution — le filtre de confiance
        # reste le même, seule la portée de la recherche s'élargit.
        if not raw:
            fallback = f"Cameroun {question}".strip()
            outcome.queries.append(fallback)
            try:
                raw = await asyncio.to_thread(self._search_sync, [fallback], limit)
            except Exception as exc:
                logger.debug("web_search_fallback_failed", error=str(exc))

        seen: set[str] = {r.url for r in outcome.results}
        for entry in raw:
            url = entry.get("href") or entry.get("url") or ""
            source = find_source(url)
            if source is None:
                outcome.discarded += 1
                continue
            if url in seen:
                continue
            seen.add(url)
            outcome.results.append(WebResult(
                title=(entry.get("title") or "").strip(),
                snippet=(entry.get("body") or "").strip()[:500],
                url=url,
                source=source,
            ))

        # Une source qui fait davantage autorité passe devant. À autorité égale,
        # une page lue intégralement devance un extrait de moteur.
        outcome.results.sort(key=lambda r: (int(r.source.priority), -len(r.snippet)))
        outcome.results = outcome.results[:limit]
        outcome.elapsed_s = round(time.perf_counter() - started, 2)

        logger.info(
            "web_search_completed",
            mode="lecture directe + moteur",
            institutions=[s.acronym for s in routing.sources],
            topics=routing.matched_topics[:5],
            pages_read=outcome.pages_read,
            kept=len(outcome.results),
            discarded=outcome.discarded,
            elapsed_s=outcome.elapsed_s,
        )
        return outcome

    def _search_sync(self, queries: list[str], limit: int) -> list[dict]:
        """
        Appel bloquant au moteur, exécuté hors de la boucle d'événements.

        On demande plus de résultats que nécessaire : le filtre de confiance en
        écarte la majeure partie, et un quota trop juste ne laisserait rien.
        """
        import warnings

        with warnings.catch_warnings():
            # La bibliothèque signale son changement de nom à chaque appel.
            warnings.simplefilter("ignore", RuntimeWarning)
            from duckduckgo_search import DDGS

            collected: list[dict] = []
            for index, query in enumerate(queries):
                # Le moteur limite le débit : deux requêtes consécutives trop
                # rapprochées reviennent vides. Un court délai entre elles suffit
                # à retrouver des résultats.
                if index:
                    time.sleep(_INTER_QUERY_DELAY_S)

                for attempt in range(_MAX_ATTEMPTS):
                    try:
                        with DDGS() as ddgs:
                            batch = ddgs.text(
                                query,
                                max_results=limit * 4,
                                region=_REGION,
                                backend="html",
                            )
                        if batch:
                            collected.extend(batch)
                            break
                        # Une réponse vide n'est pas une erreur : le moteur
                        # temporise. On lui laisse le temps avant d'abandonner.
                        if attempt + 1 < _MAX_ATTEMPTS:
                            time.sleep(_RETRY_DELAY_S * (attempt + 1))
                    except Exception as exc:
                        # Une requête qui échoue ne doit pas emporter les autres.
                        logger.debug(
                            "web_search_query_failed",
                            query=query[:60], attempt=attempt, error=str(exc),
                        )
                        if attempt + 1 < _MAX_ATTEMPTS:
                            time.sleep(_RETRY_DELAY_S * (attempt + 1))
            return collected


_service: Optional[WebSearchService] = None


def get_web_search_service() -> WebSearchService:
    global _service
    if _service is None:
        _service = WebSearchService()
    return _service


def format_for_prompt(outcome: WebSearchOutcome) -> str:
    """
    Met en forme les résultats pour le prompt.

    Les sources web sont présentées comme un bloc distinct des extraits du
    corpus : elles n'ont pas le même statut probatoire. Un extrait du corpus est
    le texte de loi lui-même ; une page web officielle en est une présentation,
    susceptible d'être datée.
    """
    if not outcome.found:
        return ""

    lines = [
        "=== SOURCES WEB OFFICIELLES (institutions camerounaises) ===",
        "Contenu lu directement sur les sites des institutions compétentes. Ces "
        "pages émanent d'organismes publics mais ne sont pas le texte normatif "
        "lui-même : elles peuvent être datées. Elles complètent le corpus, elles "
        "ne le remplacent pas. Cite-les par leur URL lorsque tu t'en sers.",
    ]
    for index, result in enumerate(outcome.results, start=1):
        stale = " [site injoignable au dernier contrôle]" if not result.source.reachable else ""
        lines.append(
            f"\n[WEB {index}] {result.source.institution} ({result.source.acronym}){stale}"
            f"\nURL: {result.url}"
            f"\nTitre: {result.title}"
            f"\n{result.snippet}"
        )
    return "\n".join(lines)
