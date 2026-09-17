"""
GOV-AI 2.0 — Vocabulaire curé des institutions et autorités camerounaises.

Le schéma Neo4j prévoyait `(Institution)-[:EMETRICE_DE]->(TexteNormatif)` mais
aucune requête n'a jamais créé un seul nœud `Institution` : le label était
déclaré, contraint, indexé — et vide. Ce module fournit la matière.

**Pourquoi une liste fermée et non une extraction automatique.** Une extraction
terminologique sur un corpus océrisé produit du bruit, et un graphe faux répond
faux avec l'assurance d'une donnée structurée. L'ensemble des organes
judiciaires et administratifs camerounais est fermé et court : l'énumérer coûte
moins cher que de filtrer les scories d'un extracteur.

**Pourquoi les termes trop généraux sont exclus.** « juge », « tribunal »,
« cour » apparaissent dans la quasi-totalité des articles d'un code de
procédure. Une institution présente partout n'indexe rien : elle ne distingue
aucun article des autres. Seules figurent ici les dénominations qui portent une
compétence identifiable — `scripts/build_knowledge_graph.py` rapporte la
sélectivité de chacune pour que ce choix reste vérifiable.

**Bilinguisme.** Le corpus contient le Code de procédure pénale en français et
le Code pénal en anglais. Les alias couvrent donc les deux langues, avec la
terminologie officielle camerounaise et non sa traduction littérale : le
« Procureur de la République » est le *State Counsel*, le « Ministère public »
est le *Legal Department*.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Institution:
    """Un organe ou une autorité dotée d'une compétence propre."""
    id: str
    label: str                       # dénomination française canonique
    label_en: str                    # dénomination anglaise officielle
    category: str                    # judiciaire | ministere | securite | politique | territorial
    aliases: tuple[str, ...] = field(default=())

    def all_forms(self) -> tuple[str, ...]:
        return (self.label, self.label_en) + self.aliases


# ── Ordre judiciaire ─────────────────────────────────────────────────────────
_JUDICIAIRE = (
    Institution(
        "procureur-republique", "Procureur de la République", "State Counsel",
        "judiciaire",
        ("procureur de la republique", "state counsel", "procureur"),
    ),
    Institution(
        "procureur-general", "Procureur Général", "Procureur General",
        "judiciaire",
        ("procureur general", "attorney general"),
    ),
    Institution(
        "ministere-public", "Ministère Public", "Legal Department",
        "judiciaire",
        ("ministere public", "legal department", "parquet"),
    ),
    Institution(
        "juge-instruction", "Juge d'instruction", "Examining Magistrate",
        "judiciaire",
        ("juge d instruction", "examining magistrate", "magistrat instructeur"),
    ),
    Institution(
        "tribunal-premiere-instance", "Tribunal de Première Instance",
        "Court of First Instance", "judiciaire",
        ("tribunal de premiere instance", "court of first instance", "tpi"),
    ),
    Institution(
        "tribunal-grande-instance", "Tribunal de Grande Instance", "High Court",
        "judiciaire",
        ("tribunal de grande instance", "high court", "tgi"),
    ),
    Institution(
        "cour-appel", "Cour d'Appel", "Court of Appeal", "judiciaire",
        ("cour d appel", "court of appeal"),
    ),
    Institution(
        "cour-supreme", "Cour Suprême", "Supreme Court", "judiciaire",
        ("cour supreme", "supreme court"),
    ),
    Institution(
        "tribunal-militaire", "Tribunal Militaire", "Military Court", "judiciaire",
        ("tribunal militaire", "military court"),
    ),
    Institution(
        "tribunal-criminel-special", "Tribunal Criminel Spécial",
        "Special Criminal Court", "judiciaire",
        ("tribunal criminel special", "special criminal court", "tcs"),
    ),
    Institution(
        "chambre-controle-instruction", "Chambre de Contrôle de l'Instruction",
        "Inquiry Control Chamber", "judiciaire",
        ("chambre de controle de l instruction", "inquiry control chamber"),
    ),
    Institution(
        "tribunal-coutumier", "Tribunal Coutumier", "Customary Court", "judiciaire",
        ("tribunal coutumier", "customary court", "juridiction traditionnelle"),
    ),
    Institution(
        "greffe", "Greffe", "Registry", "judiciaire",
        ("greffier", "greffe", "registrar", "registry"),
    ),
    Institution(
        "huissier", "Huissier de Justice", "Bailiff", "judiciaire",
        ("huissier de justice", "huissier", "bailiff"),
    ),
    Institution(
        "barreau", "Barreau", "Bar Council", "judiciaire",
        ("barreau", "bar council", "ordre des avocats"),
    ),
)

# ── Police, gendarmerie, administration pénitentiaire ────────────────────────
_SECURITE = (
    Institution(
        "officier-police-judiciaire", "Officier de Police Judiciaire",
        "Judicial Police Officer", "securite",
        ("officier de police judiciaire", "judicial police officer", "opj"),
    ),
    Institution(
        "agent-police-judiciaire", "Agent de Police Judiciaire",
        "Judicial Police Agent", "securite",
        ("agent de police judiciaire", "judicial police agent", "apj"),
    ),
    Institution(
        "police-judiciaire", "Police Judiciaire", "Judicial Police", "securite",
        ("police judiciaire", "judicial police"),
    ),
    Institution(
        "gendarmerie", "Gendarmerie Nationale", "National Gendarmerie", "securite",
        ("gendarmerie nationale", "gendarmerie", "national gendarmerie", "gendarme"),
    ),
    Institution(
        "surete-nationale", "Délégation Générale à la Sûreté Nationale",
        "General Delegation for National Security", "securite",
        ("delegation generale a la surete nationale", "surete nationale", "dgsn"),
    ),
    Institution(
        "administration-penitentiaire", "Administration Pénitentiaire",
        "Prison Administration", "securite",
        ("administration penitentiaire", "prison administration",
         "regisseur de prison", "prison superintendent"),
    ),
)

# ── Exécutif et administration ───────────────────────────────────────────────
_EXECUTIF = (
    Institution(
        "president-republique", "Président de la République",
        "President of the Republic", "politique",
        ("president de la republique", "president of the republic",
         "chef de l etat", "presidence"),
    ),
    Institution(
        "primature", "Primature", "Prime Minister's Office", "politique",
        ("primature", "premier ministre", "prime minister"),
    ),
    Institution(
        "ministere-justice", "Ministère de la Justice", "Ministry of Justice",
        "ministere",
        ("ministere de la justice", "ministry of justice", "garde des sceaux",
         "minjustice", "minister of justice", "ministre de la justice"),
    ),
    Institution(
        "ministere-defense", "Ministère de la Défense", "Ministry of Defence",
        "ministere",
        ("ministere de la defense", "ministry of defence", "mindef"),
    ),
    Institution(
        "ministere-finances", "Ministère des Finances", "Ministry of Finance",
        "ministere",
        ("ministere des finances", "ministry of finance", "minfi"),
    ),
    Institution(
        "assemblee-nationale", "Assemblée Nationale", "National Assembly", "politique",
        ("assemblee nationale", "national assembly"),
    ),
    Institution(
        "senat", "Sénat", "Senate", "politique",
        ("senat", "senate"),
    ),
    Institution(
        "conseil-constitutionnel", "Conseil Constitutionnel",
        "Constitutional Council", "politique",
        ("conseil constitutionnel", "constitutional council"),
    ),
)

# ── Administration territoriale ──────────────────────────────────────────────
_TERRITORIAL = (
    Institution(
        "prefet", "Préfet", "Senior Divisional Officer", "territorial",
        ("prefet", "prefecture", "senior divisional officer", "sdo"),
    ),
    Institution(
        "sous-prefet", "Sous-Préfet", "Divisional Officer", "territorial",
        ("sous prefet", "sous prefecture", "divisional officer"),
    ),
    Institution(
        "gouverneur", "Gouverneur", "Governor", "territorial",
        ("gouverneur", "governor"),
    ),
    Institution(
        "maire", "Maire", "Mayor", "territorial",
        ("maire", "mairie", "mayor"),
    ),
)

# ── Organisations supranationales et organismes techniques ───────────────────
_AUTRES = (
    Institution(
        "ohada", "OHADA",
        "Organisation for the Harmonisation of Business Law in Africa",
        "supranational",
        ("ohada",),
    ),
    Institution(
        "cenadi", "CENADI", "National Centre for Data Processing", "technique",
        ("cenadi", "centre national de developpement de l informatique"),
    ),
)

INSTITUTIONS: tuple[Institution, ...] = (
    _JUDICIAIRE + _SECURITE + _EXECUTIF + _TERRITORIAL + _AUTRES
)

_BY_ID = {inst.id: inst for inst in INSTITUTIONS}


def _normalize(text: str) -> str:
    """Minuscule, sans accents, ponctuation réduite à des espaces."""
    decomposed = unicodedata.normalize("NFD", (text or "").lower())
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", stripped)).strip()


def _build_matcher() -> tuple[re.Pattern[str], dict[str, str]]:
    """
    Une seule alternation, alias les plus longs d'abord.

    L'ordre importe : « procureur general » doit l'emporter sur « procureur »,
    faute de quoi toute mention du Procureur général serait imputée au Procureur
    de la République — deux compétences distinctes.

    Les bornes sont des lookarounds sur [a-z0-9] plutôt que des limites de mot :
    le texte est déjà normalisé, et cela garantit qu'un alias court comme
    « tcs » ou « opj » ne se déclenche pas à l'intérieur d'un mot.
    """
    alias_to_id: dict[str, str] = {}
    for inst in INSTITUTIONS:
        for form in inst.all_forms():
            normalized = _normalize(form)
            if normalized:
                alias_to_id.setdefault(normalized, inst.id)

    ordered = sorted(alias_to_id, key=len, reverse=True)
    pattern = re.compile(
        "(?<![a-z0-9])(?:"
        + "|".join(re.escape(alias).replace(r"\ ", r"\s+") for alias in ordered)
        + ")(?![a-z0-9])"
    )
    return pattern, alias_to_id


_MATCHER, _ALIAS_TO_ID = _build_matcher()


def find_institutions(text: str) -> list[str]:
    """
    Identifiants des institutions nommées dans un texte, sans doublon.

    L'ordre de retour suit la première apparition, ce qui laisse au lecteur du
    graphe la possibilité de privilégier l'organe mentionné en tête.
    """
    found: list[str] = []
    seen: set[str] = set()
    for match in _MATCHER.finditer(_normalize(text)):
        inst_id = _ALIAS_TO_ID.get(re.sub(r"\s+", " ", match.group(0)))
        if inst_id and inst_id not in seen:
            seen.add(inst_id)
            found.append(inst_id)
    return found


# Une question de compétence ne nomme pas l'organe qu'elle cherche — c'est
# précisément ce qu'elle demande. Ces tournures la signalent, et déclenchent la
# traversée inverse : partir des articles retrouvés pour remonter aux autorités
# qu'ils nomment.
_AUTHORITY_QUESTION_RE = re.compile(
    r"(?<![a-z0-9])(?:"
    r"quel(?:le)?s?\s+(?:autorite|organe|juridiction|instance|tribunal|service)"
    r"|qui\s+(?:peut|doit|est\s+competent|delivre|decerne|ordonne|prononce"
    r"|autorise|saisit|statue|decide)"
    r"|(?:devant|aupres\s+de|par|a)\s+qui"
    r"|est\s+competent"
    r"|which\s+(?:authority|court|body|officer|organ)"
    r"|who\s+(?:may|can|shall|must|is\s+competent|issues|orders)"
    r"|before\s+whom"
    r")(?![a-z0-9])"
)


def asks_about_authority(query: str) -> bool:
    """
    La question porte-t-elle sur une compétence ?

    Sert au routage : « quelle autorité peut décerner un mandat de détention
    provisoire » ne nomme aucune institution, donc le rapprochement par libellé
    ne donne rien — mais c'est exactement la question à laquelle le graphe
    répond mieux que la recherche vectorielle, qui se contente de ramener les
    articles traitant du mandat sans dire lequel des organes cités décide.
    """
    return bool(_AUTHORITY_QUESTION_RE.search(_normalize(query)))


def get_institution(inst_id: str) -> Optional[Institution]:
    return _BY_ID.get(inst_id)


def resolve_emitter(raw: str) -> Optional[Institution]:
    """
    Rattache l'institution émettrice saisie à l'ingestion au vocabulaire.

    Le formulaire est libre : « Ministère de la justice », « MINJUSTICE » et
    « Ministry of Justice » désignent le même organe et ne doivent pas produire
    trois nœuds.
    """
    ids = find_institutions(raw or "")
    return _BY_ID.get(ids[0]) if ids else None
