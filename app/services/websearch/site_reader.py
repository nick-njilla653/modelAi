"""
GOV-AI 2.0 — Lecture directe des sites institutionnels camerounais.

Pourquoi ce module existe : le moteur de recherche tiers limite le débit et,
même lorsqu'il répond, il ne fournit qu'un extrait de 500 caractères. Le système
ne lisait donc jamais les pages officielles — il ne faisait que citer un résumé
produit par un intermédiaire américain.

Ce module supprime l'intermédiaire. Les sites institutionnels répondent
directement depuis le serveur (mesuré : 35 domaines sur 60), et `lxml` suffit à
en extraire un texte propre — 65 Ko de HTML donnent 2 400 caractères utiles sur
impots.cm.

Le parcours est délibérément borné et respectueux :
  - une seule profondeur : page d'accueil, puis les liens dont l'intitulé
    répond à la question ;
  - jamais hors du domaine de l'institution ;
  - un budget de pages et d'octets, un délai maximal, un cache court.

Le volume impose une sélection : 355 Ko de texte ne tiennent pas dans une
fenêtre de 8192 tokens. On ne retient donc que les passages où la question
trouve un écho.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

from app.core.logging import get_logger
from app.services.websearch.official_sources import OfficialSource, _normalize

logger = get_logger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; GOV-AI/2.0; assistant administratif)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "fr,en;q=0.8",
}

# Bornes du parcours. Elles protègent autant le serveur distant que la fenêtre
# de contexte du modèle.
_PAGE_TIMEOUT_S = 8.0
_MAX_HTML_BYTES = 3_000_000
_MAX_PAGES_PER_SITE = 3
_MAX_LINKS_CONSIDERED = 60
_PASSAGE_CHARS = 1400
_CACHE_TTL_S = 900

# Un mot de la question vaut trois fois un mot du vocabulaire institutionnel :
# sans cet écart, une page « Direction Générale du Budget » devance la page qui
# traite réellement du sujet demandé.
_QUESTION_WEIGHT = 3
_INSTITUTION_WEIGHT = 1

# Au-delà de cette part de texte constituée d'intitulés de liens, la page est un
# sommaire plutôt qu'un contenu. Le facteur la fait reculer sans l'exclure : un
# sommaire vaut mieux que rien si le site n'offre pas mieux.
_LISTING_DENSITY = 0.45
_LISTING_PENALTY = 0.25
_LISTING_URL = re.compile(r"/(category|categorie|tag|rubrique|page|archives?)/", re.IGNORECASE)

# Mots trop courants pour désigner un sujet dans un intitulé de lien.
_STOPWORDS = {
    "comment", "quel", "quelle", "quels", "quelles", "pour", "avec", "dans",
    "les", "des", "une", "que", "qui", "est", "sont", "faire", "avoir", "sur",
    "par", "aux", "cameroun", "camerounais", "camerounaise", "the", "and", "for",
}

_WHITESPACE = re.compile(r"\s+")
_DROP_TAGS = ("script", "style", "noscript", "svg", "form", "iframe")


@dataclass
class PageExtract:
    """Le contenu utile d'une page officielle."""
    url: str
    title: str
    text: str
    source: OfficialSource
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.text[:500],
            "domain": self.source.domain,
            "institution": self.source.institution,
            "acronym": self.source.acronym,
            "priority": int(self.source.priority),
            "reachable": True,  # elle vient d'être lue
        }


@dataclass
class _CacheEntry:
    html: str
    fetched_at: float


_cache: dict[str, _CacheEntry] = {}


def _keywords(question: str, source: Optional[OfficialSource] = None) -> dict[str, int]:
    """
    Mots porteurs de la question, pondérés, enrichis du vocabulaire institutionnel.

    Deux besoins contraires. « Quel est le taux de TVA » ne donne que
    {taux, tva} : trop étroit pour reconnaître une page « circulaire fiscale ».
    Mais ajouter à poids égal tout le vocabulaire du MINFI — budget, douane,
    trésor — fait remonter l'organigramme du ministère plutôt que la réponse.

    D'où la pondération : les mots de la question pèsent trois fois ceux de
    l'institution. Le vocabulaire élargit sans détourner.
    """
    weights: dict[str, int] = {}
    for word in re.findall(r"[a-z0-9]{3,}", _normalize(question)):
        if word not in _STOPWORDS:
            weights[word] = _QUESTION_WEIGHT

    if source is not None:
        for topic in source.topics:
            for word in re.findall(r"[a-z0-9]{3,}", _normalize(topic)):
                if word not in _STOPWORDS:
                    weights.setdefault(word, _INSTITUTION_WEIGHT)
    return weights


def _same_site(url: str, source: OfficialSource) -> bool:
    """Vrai si l'URL reste sur le domaine de l'institution."""
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    return any(
        host == d or host.endswith("." + d)
        for d in (source.domain, *source.aliases)
    )


def extract_text(html: str) -> tuple[str, str, float]:
    """
    Extrait (titre, texte lisible, densité de liens) d'une page HTML.

    Les blocs de navigation sont retirés : ils répètent le même menu sur chaque
    page et noieraient le contenu utile dans la sélection de passages.

    La densité de liens — part du texte qui est un intitulé de lien — distingue
    une page de contenu d'une page d'index. Une catégorie « Entreprises » cite
    « impôts », « TVA », « déclaration » des dizaines de fois sans rien
    expliquer : elle domine le score par accumulation de titres, et le modèle
    reçoit un sommaire là où il attendait une procédure.
    """
    from lxml import html as lhtml

    try:
        doc = lhtml.fromstring(html)
    except Exception:
        return "", ""

    title = ""
    titles = doc.xpath("//title/text()")
    if titles:
        title = _WHITESPACE.sub(" ", str(titles[0])).strip()

    for tag in _DROP_TAGS + ("nav", "footer", "header"):
        for node in doc.xpath(f"//{tag}"):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)

    link_chars = sum(
        len(_WHITESPACE.sub(" ", a.text_content()).strip())
        for a in doc.xpath("//a")
    )
    text = _WHITESPACE.sub(" ", doc.text_content()).strip()
    density = (link_chars / len(text)) if text else 1.0
    return title, text, min(density, 1.0)


def extract_links(html: str, base_url: str, source: OfficialSource) -> list[tuple[str, str]]:
    """Liens internes (URL absolue, intitulé), dédoublonnés."""
    from lxml import html as lhtml

    try:
        doc = lhtml.fromstring(html)
    except Exception:
        return []

    seen: set[str] = set()
    links: list[tuple[str, str]] = []
    for anchor in doc.xpath("//a[@href]")[:_MAX_LINKS_CONSIDERED * 4]:
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        if not absolute.startswith(("http://", "https://")):
            continue
        if not _same_site(absolute, source):
            continue
        absolute = absolute.split("#")[0]
        if absolute in seen:
            continue
        seen.add(absolute)
        label = _WHITESPACE.sub(" ", anchor.text_content()).strip()
        links.append((absolute, label))
        if len(links) >= _MAX_LINKS_CONSIDERED:
            break
    return links


def select_passage(
    text: str, keywords: dict[str, int], width: int = _PASSAGE_CHARS
) -> tuple[str, float]:
    """
    Retient la fenêtre de texte où la question trouve le plus d'échos.

    Une page ministérielle peut peser des centaines de milliers de caractères ;
    en livrer le début serait livrer le menu d'accueil. On cherche donc l'endroit
    où les mots de la question se concentrent.
    """
    if not text:
        return "", 0.0
    if len(text) <= width:
        low = _normalize(text)
        return text, float(sum(w for k, w in keywords.items() if k in low))

    normalized = _normalize(text)
    step = max(width // 3, 200)
    best_start, best_score = 0, -1.0

    for start in range(0, len(normalized) - width + step, step):
        window = normalized[start:start + width]
        score = sum(window.count(k) * w for k, w in keywords.items())
        if score > best_score:
            best_score, best_start = score, start

    return text[best_start:best_start + width].strip(), float(max(best_score, 0))


class SiteReader:
    """Lit les sites institutionnels sans passer par un moteur tiers."""

    async def _fetch(self, client, url: str) -> Optional[str]:
        """Récupère une page, avec cache court et plafond de taille."""
        cached = _cache.get(url)
        if cached and time.time() - cached.fetched_at < _CACHE_TTL_S:
            return cached.html

        try:
            response = await client.get(url, timeout=_PAGE_TIMEOUT_S, follow_redirects=True)
        except Exception as exc:
            logger.debug("site_fetch_failed", url=url[:90], error=type(exc).__name__)
            return None

        if response.status_code >= 400:
            logger.debug("site_fetch_status", url=url[:90], status=response.status_code)
            return None
        if "html" not in response.headers.get("content-type", "").lower():
            return None
        if len(response.content) > _MAX_HTML_BYTES:
            return None

        _cache[url] = _CacheEntry(html=response.text, fetched_at=time.time())
        return response.text

    async def read_institution(
        self,
        source: OfficialSource,
        question: str,
        max_pages: int = _MAX_PAGES_PER_SITE,
    ) -> list[PageExtract]:
        """
        Lit le site d'une institution en suivant les liens qui répondent à la
        question, puis rend les passages pertinents.
        """
        import httpx

        keywords = _keywords(question, source)
        extracts: list[PageExtract] = []

        async with httpx.AsyncClient(headers=_HEADERS) as client:
            home_url = f"https://{source.domain}"
            html = await self._fetch(client, home_url)
            if html is None:
                html = await self._fetch(client, f"https://www.{source.domain}")
                if html is None:
                    return []
                home_url = f"https://www.{source.domain}"

            title, text, _ = extract_text(html)
            home_passage, home_score = select_passage(text, keywords)

            # Les liens dont l'intitulé fait écho à la question passent devant.
            scored_links: list[tuple[float, str, str]] = []
            for url, label in extract_links(html, home_url, source):
                haystack = _normalize(f"{label} {url}")
                hits = sum(w for k, w in keywords.items() if k in haystack)
                if hits:
                    scored_links.append((hits, url, label))
            scored_links.sort(key=lambda item: item[0], reverse=True)

            for _, url, label in scored_links[:max_pages]:
                page_html = await self._fetch(client, url)
                if page_html is None:
                    continue
                page_title, page_text, density = extract_text(page_html)
                passage, score = select_passage(page_text, keywords)
                if not passage:
                    continue

                # Une page dont l'essentiel du texte est fait de liens est un
                # sommaire : son score vient de l'accumulation de titres, pas
                # d'un contenu qui répond.
                if density > _LISTING_DENSITY:
                    score *= _LISTING_PENALTY
                if _LISTING_URL.search(url):
                    score *= _LISTING_PENALTY

                extracts.append(
                    PageExtract(url, page_title or label, passage, source, score)
                )

        # La page d'accueil n'est retenue que si aucune page ciblée n'a été
        # trouvée : elle contient surtout des menus et des actualités.
        if not extracts and home_passage:
            extracts.append(PageExtract(home_url, title, home_passage, source, home_score))

        extracts.sort(key=lambda e: e.score, reverse=True)
        logger.info(
            "site_read",
            institution=source.acronym,
            pages=len(extracts),
            best_score=extracts[0].score if extracts else 0,
        )
        return extracts

    async def read_institutions(
        self,
        sources: list[OfficialSource],
        question: str,
        max_pages_total: int = 4,
    ) -> list[PageExtract]:
        """
        Lit plusieurs institutions en parallèle et retient les meilleurs passages.

        Le parallélisme est nécessaire : chaque site demande plusieurs secondes,
        et les lire en série ajouterait une minute à une réponse qui en prend
        déjà deux.
        """
        started = time.perf_counter()
        # Les domaines connus injoignables sont écartés de la lecture directe :
        # les attendre coûterait le délai complet pour rien. Ils restent en
        # revanche routables pour la recherche par moteur, qui peut en servir
        # une page indexée.
        readable = [s for s in sources if s.reachable]
        skipped = [s.acronym for s in sources if not s.reachable]
        if skipped:
            logger.debug("sites_skipped_unreachable", institutions=skipped)

        results = await asyncio.gather(
            *(self.read_institution(s, question) for s in readable),
            return_exceptions=True,
        )

        extracts: list[PageExtract] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.debug("site_read_failed", error=str(result)[:120])
                continue
            extracts.extend(result)

        # Un passage sans écho de la question n'apporte rien : c'est le menu
        # d'accueil du site.
        extracts = [e for e in extracts if e.score > 0]
        extracts.sort(key=lambda e: (e.score, -int(e.source.priority)), reverse=True)

        logger.info(
            "sites_read",
            institutions=[s.acronym for s in readable],
            kept=min(len(extracts), max_pages_total),
            elapsed_s=round(time.perf_counter() - started, 2),
        )
        return extracts[:max_pages_total]


_reader: Optional[SiteReader] = None


def get_site_reader() -> SiteReader:
    global _reader
    if _reader is None:
        _reader = SiteReader()
    return _reader
