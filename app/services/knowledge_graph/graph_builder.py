"""
GOV-AI 2.0 — Peuplement du graphe de connaissances juridiques.

Le graphe était scindé en deux moitiés dont une seule existait : le schéma était
créé (`ensure_schema`), le lecteur écrit (`KnowledgeGraphService`), et les
messages d'ingestion annonçaient « KG + Milvus + ES » — mais aucune requête
d'écriture n'existait dans le projet. Neo4j est resté vide, et
`enrich_context` retournait systématiquement rien.

Ce module écrit la moitié manquante. Il produit exactement la forme que le
lecteur interroge :

    (t:TexteNormatif {id, title, doc_type, jurisdiction})
    (a:Article {id, number, title, content_preview})
    (i:Institution {id, label, label_en, category})
    (t)-[:A_ARTICLE]->(a)
    (a)-[:REFERENCE]->(autre:Article)
    (i)-[:EMETRICE_DE]->(t)
    (a)-[:MENTIONNE]->(i)

Le prérequis n'existait qu'après la persistance de `article_ref` : sans numéro
d'article attaché à chaque passage, il n'y avait littéralement rien à mettre
dans le graphe.

**Portée** : le corpus global uniquement. Une pièce jointe de conversation ne
doit jamais atteindre le graphe, qui est partagé par toutes les conversations.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.logging import get_logger
from app.services.knowledge_graph.institutions import (
    find_institutions,
    get_institution,
    resolve_emitter,
)

logger = get_logger(__name__)

# Renvoi vers un article, éventuellement suivi du texte cité :
#   « article 157 (1) », « articles 93 à 99 », « article 169 du Code Pénal »
# Textes normatifs citables. Une capture ouverte (« Code\s+[^,.]{3,48} ») avalait
# la suite de la phrase : « Code Pénal est passible des peines prévues à l'articl »
# devenait un titre de texte, et le graphe se serait peuplé de nœuds de charabia.
# Une liste fermée ne reconnaît que ce qui est réellement un texte ; un renvoi
# vers un texte inconnu n'est pas rattaché plutôt que mal rattaché.
_CITED_TEXTS = (
    "Code de Procédure Pénale", "Code de Procedure Penale",
    "Criminal Procedure Code", "Penal Code",
    "Code Pénal", "Code Penal",
    "Code de Procédure Civile", "Code Civil",
    "Code du Travail", "Code Général des Impôts", "Code General des Impots",
    "Code de la Famille", "Code Forestier", "Code Minier",
    "Code des Douanes", "Code Électoral", "Code Electoral",
    "Constitution",
)

_CITED_TEXT_ALT = "|".join(
    re.escape(name).replace(r"\ ", r"\s+") for name in
    sorted(_CITED_TEXTS, key=len, reverse=True)   # le plus long d'abord
)

# Renvoi vers un article, éventuellement suivi du texte cité :
#   « article 157 (1) », « articles 93 à 99 », « article 169 du Code Pénal »
_REFERENCE_RE = re.compile(
    # « article » en français, « section » en anglais : le Code pénal du corpus
    # est la version anglaise et dit « Section 169 », non « article 169 ».
    r"\b(?:articles?|sections?)\s+"
    r"(\d{1,4}(?:\s*(?:bis|ter|quater))?)"          # 1 : numéro cité
    r"(?:\s*\([^)]{1,8}\))?"                        # alinéa éventuel, ignoré
    r"(?:\s*(?:à|a|et|and|to|,)\s*(\d{1,4}))?"      # 2 : borne d'une plage
    r"(?:\s+(?:du|de\s+la|de\s+l['’]|des|of\s+the|of)\s*"
    rf"({_CITED_TEXT_ALT}))?",                       # 3 : texte cité, liste fermée
    re.IGNORECASE,
)

# L'en-tête d'un article ne doit pas être lu comme un renvoi à lui-même.
_LEADING_HEADER_RE = re.compile(
    r"^\s*(?:Article|Section)\s+\d{1,4}(?:\s*(?:bis|ter|quater))?\s*[:.\-—]*\s*",
    re.IGNORECASE,
)

# Au-delà, une « plage » d'articles n'en est plus une : c'est une coïncidence de
# chiffres, et matérialiser 400 renvois fictifs polluerait le graphe.
_MAX_RANGE_SPAN = 20

# Un même texte se cite sous plusieurs noms, et d'une langue à l'autre. Sans
# cette table, « Penal Code » et « Code Pénal » créeraient deux nœuds fantômes
# à côté du document réel — et le Code pénal, qui se cite lui-même par son nom,
# se dédoublerait.
_TEXT_ALIASES: dict[str, tuple[str, ...]] = {
    "code penal": ("penal code", "code penal"),
    "penal code": ("penal code", "code penal"),
    "code de procedure penale": ("criminal procedure code", "code de procedure penale"),
    "criminal procedure code": ("criminal procedure code", "code de procedure penale"),
}


def _resolve_cited_text(cited: str, corpus: dict[str, str]) -> Optional[str]:
    """
    Rattache un texte cité à un document du corpus, s'il s'y trouve.

    `corpus` associe le titre normalisé de chaque document à son identifiant.
    Le rattachement rend explicite le lien entre codes — « l'article 169 du Code
    Pénal » cité par le Code de procédure pénale pointe alors vers la véritable
    disposition, au lieu d'un nœud vide.
    """
    needle = _normalize(cited)
    candidates = _TEXT_ALIASES.get(needle, (needle,))
    for title, doc_id in corpus.items():
        if any(alias in title or title.startswith(alias) for alias in candidates):
            return doc_id
    return None

@dataclass
class GraphStats:
    """Ce que le peuplement a effectivement écrit."""
    texts: int = 0
    articles: int = 0
    belongs_to: int = 0
    references: int = 0
    institutions: int = 0
    mentions: int = 0
    emitter: Optional[str] = None
    external_texts: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "texts": self.texts,
            "articles": self.articles,
            "belongs_to": self.belongs_to,
            "references": self.references,
            "institutions": self.institutions,
            "mentions": self.mentions,
            "emitter": self.emitter,
            "external_texts": sorted(self.external_texts),
        }


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", stripped)).strip()


def _text_key(title: str) -> str:
    """Identifiant stable d'un texte normatif, dérivé de son intitulé."""
    return _normalize(title).replace(" ", "-")[:80] or "texte-inconnu"


def article_number(article_ref: str) -> Optional[str]:
    """
    Extrait le numéro d'une référence (« Article 103 » → « 103 »).

    Les continuations marquées « (suite) » sont écartées : elles désignent le
    même article que le passage précédent, et en créer un nœud dupliquerait la
    disposition.

    Publique parce que l'identifiant d'un nœud `Article` est « {doc_id}:{numéro} » :
    quiconque veut interroger le graphe depuis un passage retrouvé doit
    reconstituer ce numéro exactement comme le peuplement l'a fait. Deux
    implémentations divergentes donneraient des identifiants qui ne se
    rejoignent pas, et la traversée retournerait vide sans erreur.
    """
    if not article_ref or "(suite)" in article_ref or "(cont.)" in article_ref:
        return None
    match = re.search(r"(\d{1,4}(?:[-\s]\d{1,4})?(?:\s*(?:bis|ter|quater))?)", article_ref)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else None


#: Nom historique, conservé pour les appels internes et les tests existants.
_article_number = article_number


def extract_references(content: str) -> list[tuple[str, Optional[str]]]:
    """
    Relève les renvois d'un passage : (numéro cité, texte cité ou None).

    Un renvoi sans texte nommé vise le document courant ; un renvoi nommé —
    « à l'article 169 du Code Pénal » — vise un autre texte. Cette distinction
    est précisément celle qui manquait au tout début : le modèle prenait les
    renvois sortants du Code de procédure pénale pour des articles de ce code.
    """
    found: list[tuple[str, Optional[str]]] = []
    seen: set[tuple[str, Optional[str]]] = set()

    # « Article 100 : L'inobservation… » commence par sa propre référence.
    body = _LEADING_HEADER_RE.sub("", content or "")

    for match in _REFERENCE_RE.finditer(body):
        first, last, cited_text = match.group(1), match.group(2), match.group(3)
        first = re.sub(r"\s+", " ", first).strip()

        target_text = re.sub(r"\s+", " ", cited_text).strip(" .,;:") if cited_text else None

        numbers = [first]
        if last:
            try:
                start, end = int(re.sub(r"\D", "", first) or 0), int(last)
                if 0 < end - start <= _MAX_RANGE_SPAN:
                    numbers = [str(n) for n in range(start, end + 1)]
            except ValueError:
                pass

        for number in numbers:
            key = (number, target_text)
            if key not in seen:
                seen.add(key)
                found.append(key)

    return found


async def build_document_graph(
    doc_id: str,
    title: str,
    doc_type: Optional[str],
    jurisdiction: Optional[str],
    chunks: list[Any],
    corpus_titles: Optional[dict[str, str]] = None,
    institution: Optional[str] = None,
) -> GraphStats:
    """
    Écrit dans Neo4j le texte, ses articles, leurs renvois et les institutions.

    Idempotent : `MERGE` partout, de sorte qu'une ré-ingestion du même document
    n'empile pas de doublons.

    Args:
        chunks: passages porteurs de `article_ref` et `content`.
        institution: institution émettrice saisie à l'ingestion (champ libre).
    """
    from app.storage.neo4j_client import run_query

    stats = GraphStats()
    # Titres normalisés des documents du corpus, pour résoudre les renvois
    # sortants vers de vrais articles plutôt que vers des nœuds vides.
    corpus = {_normalize(k): v for k, v in (corpus_titles or {}).items()}
    corpus[_normalize(title)] = doc_id

    # ── Le texte normatif ─────────────────────────────────────────────────────
    await run_query(
        """
        MERGE (t:TexteNormatif {id: $id})
        SET t.title = $title,
            t.doc_type = $doc_type,
            t.jurisdiction = $jurisdiction,
            t.in_corpus = true
        """,
        {
            "id": doc_id,
            "title": title,
            "doc_type": doc_type or "autre",
            "jurisdiction": jurisdiction or "national",
        },
    )
    stats.texts = 1

    # ── L'institution émettrice ───────────────────────────────────────────────
    # Elle est déjà saisie au formulaire d'ingestion ; il ne manquait que
    # l'arête. Le champ étant libre, il passe par le vocabulaire pour que
    # « MINJUSTICE » et « Ministère de la justice » ne fassent qu'un nœud.
    emitter = resolve_emitter(institution) if institution else None
    if emitter:
        await run_query(
            """
            MERGE (i:Institution {id: $id})
            SET i.label = $label, i.label_en = $label_en, i.category = $category
            WITH i
            MATCH (t:TexteNormatif {id: $doc_id})
            MERGE (i)-[:EMETRICE_DE]->(t)
            """,
            {
                "id": emitter.id,
                "label": emitter.label,
                "label_en": emitter.label_en,
                "category": emitter.category,
                "doc_id": doc_id,
            },
        )
        stats.emitter = emitter.id
    elif institution:
        # Trace explicite : une institution saisie mais hors vocabulaire est un
        # signal utile, soit que la liste est à compléter, soit que la saisie
        # est fautive. La taire reviendrait à perdre l'information.
        logger.info("graph_emitter_unknown", doc_id=doc_id, institution=institution[:60])

    # ── Les articles ──────────────────────────────────────────────────────────
    # Un article peut couvrir plusieurs passages : on retient le premier, et le
    # plus long aperçu rencontré.
    articles: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        number = _article_number(getattr(chunk, "article_ref", "") or "")
        if not number:
            continue
        content = (getattr(chunk, "content", "") or "").strip()
        entry = articles.setdefault(
            number, {"preview": "", "page": None, "refs": [], "institutions": []}
        )
        if len(content) > len(entry["preview"]):
            entry["preview"] = content[:400]
        if entry["page"] is None:
            entry["page"] = getattr(chunk, "page", None)
        entry["refs"].extend(extract_references(content))
        # Un article s'étale parfois sur plusieurs passages : les institutions
        # sont relevées sur chacun, pas seulement sur l'aperçu retenu.
        entry["institutions"].extend(find_institutions(content))

    if not articles:
        logger.warning("graph_no_articles", doc_id=doc_id, title=title)
        return stats

    await run_query(
        """
        UNWIND $rows AS row
        MATCH (t:TexteNormatif {id: $doc_id})
        MERGE (a:Article {id: row.id})
        SET a.number = row.number,
            a.title = row.title,
            a.content_preview = row.preview,
            a.page = row.page
        MERGE (t)-[:A_ARTICLE]->(a)
        """,
        {
            "doc_id": doc_id,
            "rows": [
                {
                    "id": f"{doc_id}:{number}",
                    "number": number,
                    "title": f"Article {number}",
                    "preview": data["preview"],
                    "page": data["page"],
                }
                for number, data in articles.items()
            ],
        },
    )
    stats.articles = len(articles)
    stats.belongs_to = len(articles)

    # ── Les renvois ───────────────────────────────────────────────────────────
    internal: list[dict[str, str]] = []
    external: list[dict[str, str]] = []

    for number, data in articles.items():
        source_id = f"{doc_id}:{number}"
        for target_number, cited_text in data["refs"]:
            if cited_text is None:
                # Renvoi interne : il ne vaut que si l'article visé existe.
                if target_number in articles and target_number != number:
                    internal.append({
                        "from": source_id,
                        "to": f"{doc_id}:{target_number}",
                    })
            else:
                resolved = _resolve_cited_text(cited_text, corpus)
                if resolved:
                    # Le texte cité est au corpus : on pointe vers sa véritable
                    # disposition. Un renvoi vers soi-même est ignoré.
                    if not (resolved == doc_id and target_number == number):
                        internal.append({
                            "from": source_id,
                            "to": f"{resolved}:{target_number}",
                        })
                    continue

                key = _text_key(cited_text)
                external.append({
                    "from": source_id,
                    "text_id": f"ext:{key}",
                    "text_title": cited_text,
                    "to": f"ext:{key}:{target_number}",
                    "number": target_number,
                })
                stats.external_texts.add(cited_text)

    if internal:
        # MATCH et non MERGE sur la cible : un renvoi vers un article qui
        # n'existe pas dans le corpus ne doit pas créer de disposition vide.
        await run_query(
            """
            UNWIND $rows AS row
            MATCH (a:Article {id: row.from})
            MATCH (b:Article {id: row.to})
            MERGE (a)-[:REFERENCE]->(b)
            """,
            {"rows": internal},
        )
        stats.references += len(internal)

    if external:
        # Un texte cité mais absent du corpus devient un nœud marqué comme tel :
        # le graphe doit pouvoir dire « ce code renvoie à un texte que je n'ai
        # pas », plutôt que de laisser le renvoi disparaître.
        await run_query(
            """
            UNWIND $rows AS row
            MATCH (a:Article {id: row.from})
            MERGE (t:TexteNormatif {id: row.text_id})
              ON CREATE SET t.title = row.text_title, t.in_corpus = false
            MERGE (b:Article {id: row.to})
              ON CREATE SET b.number = row.number,
                            b.title = 'Article ' + row.number,
                            b.content_preview = ''
            MERGE (t)-[:A_ARTICLE]->(b)
            MERGE (a)-[:REFERENCE]->(b)
            """,
            {"rows": external},
        )
        stats.references += len(external)

    # ── Les institutions citées dans les articles ─────────────────────────────
    # C'est la moitié qui vaut quelque chose : « quelle autorité peut décerner
    # un mandat de détention provisoire » devient une traversée de graphe, là
    # où la recherche vectorielle se contente de ramener les articles qui
    # *parlent* du mandat sans dire lequel des organes cités est compétent.
    mentions = [
        {"article_id": f"{doc_id}:{number}", "institution_id": inst_id}
        for number, data in articles.items()
        for inst_id in dict.fromkeys(data["institutions"])
    ]
    if mentions:
        cited = {m["institution_id"] for m in mentions}
        await run_query(
            """
            UNWIND $rows AS row
            MERGE (i:Institution {id: row.id})
            SET i.label = row.label, i.label_en = row.label_en, i.category = row.category
            """,
            {
                "rows": [
                    {
                        "id": inst.id,
                        "label": inst.label,
                        "label_en": inst.label_en,
                        "category": inst.category,
                    }
                    for inst in (get_institution(i) for i in sorted(cited))
                    if inst is not None
                ]
            },
        )
        await run_query(
            """
            UNWIND $rows AS row
            MATCH (a:Article {id: row.article_id})
            MATCH (i:Institution {id: row.institution_id})
            MERGE (a)-[:MENTIONNE]->(i)
            """,
            {"rows": mentions},
        )
        stats.institutions = len(cited)
        stats.mentions = len(mentions)

    logger.info(
        "graph_document_built",
        doc_id=doc_id,
        title=title[:60],
        **{k: v for k, v in stats.to_dict().items() if k != "external_texts"},
        external_texts=len(stats.external_texts),
    )
    return stats


async def delete_document_graph(doc_id: str) -> None:
    """Retire un document du graphe, avec ses articles propres."""
    from app.storage.neo4j_client import run_query

    await run_query(
        """
        MATCH (t:TexteNormatif {id: $doc_id})-[:A_ARTICLE]->(a:Article)
        DETACH DELETE a
        """,
        {"doc_id": doc_id},
    )
    await run_query(
        "MATCH (t:TexteNormatif {id: $doc_id}) DETACH DELETE t",
        {"doc_id": doc_id},
    )
    # Les institutions sont partagées entre documents : on ne les supprime que
    # lorsqu'elles ne sont plus rattachées à rien, sinon retirer un texte
    # amputerait le graphe des autres.
    await run_query(
        "MATCH (i:Institution) WHERE NOT (i)--() DELETE i"
    )
    logger.info("graph_document_deleted", doc_id=doc_id)


async def graph_stats() -> dict:
    """Volumétrie du graphe, pour la page Santé et la vérification."""
    from app.storage.neo4j_client import run_query

    rows = await run_query(
        """
        OPTIONAL MATCH (t:TexteNormatif) WITH count(t) AS texts
        OPTIONAL MATCH (a:Article) WITH texts, count(a) AS articles
        OPTIONAL MATCH ()-[r:REFERENCE]->() WITH texts, articles, count(r) AS refs
        OPTIONAL MATCH (e:TexteNormatif {in_corpus: false})
        WITH texts, articles, refs, count(e) AS external_texts
        OPTIONAL MATCH (i:Institution)
        WITH texts, articles, refs, external_texts, count(i) AS institutions
        OPTIONAL MATCH ()-[m:MENTIONNE]->()
        RETURN texts, articles, refs, external_texts, institutions,
               count(m) AS mentions
        """
    )
    return rows[0] if rows else {
        "texts": 0, "articles": 0, "refs": 0,
        "external_texts": 0, "institutions": 0, "mentions": 0,
    }
