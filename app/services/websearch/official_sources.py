"""
GOV-AI 2.0 — Registre des sources officielles camerounaises.

Ce module remplit deux rôles distincts, et c'est leur séparation qui fait la
valeur du dispositif :

1. **Routage** — une question porte sur un domaine (fiscalité, foncier, état
   civil…) dont relèvent des institutions précises. Nommer ces institutions dans
   la requête est ce qui fait remonter les sites officiels : une mesure sur le
   moteur montre que « MINFI Cameroun taux TVA » ramène 7 résultats en .cm sur 8,
   là où « Cameroun création entreprise » n'en ramène aucun.

2. **Filtre de confiance** — un résultat hors de ce registre n'est jamais
   présenté comme officiel. L'opérateur `site:` du moteur s'avérant peu fiable,
   le filtrage se fait sur les résultats et non sur la requête : c'est le seul
   point de contrôle qui ne dépende pas du comportement du moteur.

Le domaine .gov.cm est celui prescrit aux sites gouvernementaux, mais nombre
d'institutions publiques utilisent un .cm simple (prc.cm, impots.cm, senat.cm) :
s'en tenir à .gov.cm écarterait la Présidence et la Direction générale des impôts.

Ce registre est une liste blanche initiale, non un inventaire juridiquement
exhaustif des domaines de l'État. Les domaines ont été relevés puis vérifiés
contre l'annuaire officiel des Services du Premier Ministre.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class Priority(IntEnum):
    """
    Niveau de confiance d'une source. Plus la valeur est basse, plus la source
    fait autorité — l'ordre sert au classement des résultats.
    """
    INSTITUTION_SUPREME = 0   # Présidence, Primature, Parlement, portail des services
    MINISTERE = 1             # Ministères sectoriels
    ADMINISTRATION = 2        # Administrations et agences spécialisées
    ORGANISME_PUBLIC = 3      # Entreprises et organismes publics
    INTERNATIONAL = 4         # Organisations régionales dont le Cameroun est membre
    PRESSE_PUBLIQUE = 5       # Médias d'État : actualité, jamais valeur normative


@dataclass(frozen=True)
class OfficialSource:
    """Une source officielle et les sujets dont elle est compétente."""
    domain: str
    institution: str
    acronym: str
    priority: Priority
    topics: tuple[str, ...]
    #: Termes injectés dans la requête pour faire remonter cette source.
    search_terms: tuple[str, ...] = ()
    #: Autres domaines de la même institution (services en ligne, portails).
    aliases: tuple[str, ...] = ()
    note: str = ""

    @property
    def query_hint(self) -> str:
        """Fragment de requête désignant l'institution."""
        return " ".join(self.search_terms) if self.search_terms else self.acronym

    @property
    def reachable(self) -> bool:
        """
        Faux si le domaine ne répondait pas au dernier sondage.

        N'exclut PAS la source : une page indexée par le moteur reste
        authentique même si le site ne répond pas à notre sonde. Ce drapeau sert
        à annoter le résultat et à éviter une tentative de lecture directe.
        """
        return self.domain not in UNREACHABLE_AT_LAST_PROBE


# Relevé de disponibilité du 31 août 2026, effectué depuis l'hébergement du
# service. Un quart des sites publics camerounais n'a pas répondu : certains
# filtrent les requêtes extérieures, d'autres sont hors ligne par intermittence.
# Plusieurs figurent pourtant à l'annuaire officiel des Services du Premier
# Ministre — leur silence ne remet pas en cause leur légitimité.
UNREACHABLE_AT_LAST_PROBE: frozenset[str] = frozenset({
    "servicepublic.gov.cm", "assemblenationale.cm", "minesec.cm", "minedub.gov.cm",
    "minefop.gov.cm", "mintss.gov.cm", "minfof.cm", "minepat.cm", "minpmeesa.gov.cm",
    "minmidt.cm", "minmap.gov.cm", "mindaf.gov.cm", "minduh.gov.cm", "mintp.gov.cm",
    "minee.gov.cm", "mint.gov.cm", "minas.gov.cm", "minjeun.gov.cm", "elecam.cm",
    "conac.cm", "cndhl.cm", "ins-cameroun.org", "api.gov.cm", "eneo.cm", "caa.cm",
})


# ── Le registre ───────────────────────────────────────────────────────────────

SOURCES: tuple[OfficialSource, ...] = (
    # ── P0 — Institutions suprêmes ────────────────────────────────────────────
    OfficialSource(
        "prc.cm", "Présidence de la République", "PRC", Priority.INSTITUTION_SUPREME,
        ("decret", "nomination", "discours", "presidence", "chef de l'etat"),
        ("Présidence de la République Cameroun", "prc.cm"),
    ),
    OfficialSource(
        "spm.gov.cm", "Services du Premier Ministre", "SPM", Priority.INSTITUTION_SUPREME,
        ("decret", "arrete", "communique", "gouvernement", "politique publique"),
        ("Services du Premier Ministre Cameroun", "spm.gov.cm"),
    ),
    OfficialSource(
        "servicepublic.gov.cm", "Portail des services publics", "ServicePublic",
        Priority.INSTITUTION_SUPREME,
        ("demarche", "procedure", "formulaire", "document administratif", "guichet"),
        ("servicepublic.gov.cm démarches administratives Cameroun",),
        note="Certificat TLS non vérifiable au sondage — contenu accessible mais non authentifié.",
    ),
    OfficialSource(
        "assemblenationale.cm", "Assemblée Nationale", "ASSNAT", Priority.INSTITUTION_SUPREME,
        ("loi", "proposition de loi", "debat parlementaire", "depute"),
        ("Assemblée Nationale Cameroun",),
    ),
    OfficialSource(
        "senat.cm", "Sénat", "SENAT", Priority.INSTITUTION_SUPREME,
        ("loi", "senateur", "parlement"),
        ("Sénat Cameroun",),
    ),

    # ── P1 — Ministères ───────────────────────────────────────────────────────
    OfficialSource(
        "minjustice.gov.cm", "Ministère de la Justice", "MINJUSTICE", Priority.MINISTERE,
        ("justice", "loi", "tribunal", "audience", "casier judiciaire", "droit",
         "penal", "procedure penale", "avocat", "detention"),
        ("MINJUSTICE Cameroun ministère de la justice",),
    ),
    OfficialSource(
        "minfi.gov.cm", "Ministère des Finances", "MINFI", Priority.MINISTERE,
        ("impot", "fiscalite", "budget", "tva", "tresor", "solde", "finances",
         "loi de finances", "douane", "declaration"),
        ("MINFI Cameroun ministère des finances",),
        aliases=("minfi.cm",),  # ebulletin.minfi.cm, ebon.minfi.cm
    ),
    OfficialSource(
        "minat.gov.cm", "Ministère de l'Administration Territoriale", "MINAT", Priority.MINISTERE,
        ("administration territoriale", "association", "prefecture", "carte nationale",
         "etat civil", "securite civile", "chefferie"),
        ("MINAT Cameroun administration territoriale",),
    ),
    OfficialSource(
        "minfopra.gov.cm", "Ministère de la Fonction Publique", "MINFOPRA", Priority.MINISTERE,
        ("concours", "fonction publique", "fonctionnaire", "carriere", "recrutement",
         "avancement", "retraite fonctionnaire"),
        ("MINFOPRA Cameroun fonction publique concours",),
    ),
    OfficialSource(
        "diplocam.cm", "Ministère des Relations Extérieures", "MINREX", Priority.MINISTERE,
        ("passeport", "visa", "consulat", "legalisation", "diaspora", "ambassade",
         "diplomatie", "etranger"),
        ("MINREX Diplocam Cameroun affaires consulaires",),
    ),
    OfficialSource(
        "minsante.gov.cm", "Ministère de la Santé Publique", "MINSANTE", Priority.MINISTERE,
        ("sante", "hopital", "vaccin", "medicament", "epidemie", "soin",
         "medecin", "pharmacie", "maladie"),
        ("MINSANTE Cameroun ministère santé publique",),
    ),
    OfficialSource(
        "minesup.gov.cm", "Ministère de l'Enseignement Supérieur", "MINESUP", Priority.MINISTERE,
        ("universite", "enseignement superieur", "diplome", "etudiant", "bourse",
         "equivalence", "grande ecole", "enspy", "inscription"),
        ("MINESUP Cameroun enseignement supérieur",),
    ),
    OfficialSource(
        "minesec.cm", "Ministère des Enseignements Secondaires", "MINESEC", Priority.MINISTERE,
        ("lycee", "college", "baccalaureat", "probatoire", "bepc", "enseignement secondaire"),
        ("MINESEC Cameroun enseignements secondaires",),
    ),
    OfficialSource(
        "minedub.gov.cm", "Ministère de l'Éducation de Base", "MINEDUB", Priority.MINISTERE,
        ("ecole primaire", "education de base", "cep", "maternelle", "instituteur"),
        ("MINEDUB Cameroun éducation de base",),
    ),
    OfficialSource(
        "minefop.gov.cm", "Ministère de l'Emploi et de la Formation Professionnelle",
        "MINEFOP", Priority.MINISTERE,
        ("emploi", "formation professionnelle", "apprentissage", "stage", "metier",
         "insertion"),
        ("MINEFOP Cameroun emploi formation professionnelle",),
    ),
    OfficialSource(
        "mintss.gov.cm", "Ministère du Travail et de la Sécurité Sociale", "MINTSS",
        Priority.MINISTERE,
        ("travail", "contrat de travail", "licenciement", "conge", "salaire",
         "smig", "syndicat", "securite sociale", "employeur", "code du travail"),
        ("MINTSS Cameroun travail sécurité sociale",),
    ),
    OfficialSource(
        "minader.cm", "Ministère de l'Agriculture et du Développement Rural", "MINADER",
        Priority.MINISTERE,
        ("agriculture", "agricole", "semence", "engrais", "culture", "phytosanitaire",
         "developpement rural", "campagne agricole"),
        ("MINADER Cameroun agriculture développement rural",),
    ),
    OfficialSource(
        "minepia.cm", "Ministère de l'Élevage, des Pêches et des Industries Animales",
        "MINEPIA", Priority.MINISTERE,
        ("elevage", "peche", "betail", "veterinaire", "industrie animale", "aquaculture"),
        ("MINEPIA Cameroun élevage pêches",),
    ),
    OfficialSource(
        "minep.gov.cm", "Ministère de l'Environnement et du Développement Durable",
        "MINEPDED", Priority.MINISTERE,
        ("environnement", "pollution", "biodiversite", "developpement durable",
         "etude d'impact", "climat", "dechet"),
        ("MINEPDED Cameroun environnement développement durable",),
    ),
    OfficialSource(
        "minfof.cm", "Ministère des Forêts et de la Faune", "MINFOF", Priority.MINISTERE,
        ("foret", "faune", "bois", "chasse", "aire protegee", "exploitation forestiere"),
        ("MINFOF Cameroun forêts faune",),
    ),
    OfficialSource(
        "minepat.cm", "Ministère de l'Économie et de la Planification", "MINEPAT",
        Priority.MINISTERE,
        ("economie", "planification", "amenagement du territoire", "investissement public",
         "plan de developpement"),
        ("MINEPAT Cameroun économie planification",),
    ),
    OfficialSource(
        "mincommerce.gov.cm", "Ministère du Commerce", "MINCOMMERCE", Priority.MINISTERE,
        ("commerce", "prix", "importation", "exportation", "concurrence", "consommateur",
         "registre du commerce", "fonds de commerce", "entreprise"),
        ("MINCOMMERCE Cameroun ministère du commerce",),
    ),
    OfficialSource(
        "minpmeesa.gov.cm", "Ministère des PME, de l'Économie Sociale et de l'Artisanat",
        "MINPMEESA", Priority.MINISTERE,
        ("pme", "artisanat", "economie sociale", "entreprise", "cooperative",
         "creation d'entreprise", "creer une entreprise", "auto-emploi"),
        ("MINPMEESA Cameroun PME artisanat",),
    ),
    OfficialSource(
        "minmidt.cm", "Ministère des Mines, de l'Industrie et du Développement Technologique",
        "MINMIDT", Priority.MINISTERE,
        ("mine", "industrie", "carriere miniere", "permis minier", "technologie"),
        ("MINMIDT Cameroun mines industrie",),
    ),
    OfficialSource(
        "minmap.gov.cm", "Ministère des Marchés Publics", "MINMAP", Priority.MINISTERE,
        ("marche public", "appel d'offres", "adjudication", "commande publique",
         "soumissionnaire"),
        ("MINMAP Cameroun marchés publics",),
    ),
    OfficialSource(
        "mindaf.gov.cm", "Ministère des Domaines, du Cadastre et des Affaires Foncières",
        "MINDCAF", Priority.MINISTERE,
        ("titre foncier", "terrain", "cadastre", "domaine", "foncier", "immatriculation",
         "concession", "expropriation", "bornage"),
        ("MINDCAF Cameroun domaines cadastre affaires foncières",),
    ),
    OfficialSource(
        "minduh.gov.cm", "Ministère de l'Habitat et du Développement Urbain", "MINHDU",
        Priority.MINISTERE,
        ("habitat", "urbanisme", "permis de construire", "logement", "voirie urbaine",
         "lotissement"),
        ("MINHDU Cameroun habitat développement urbain",),
    ),
    OfficialSource(
        "mintp.gov.cm", "Ministère des Travaux Publics", "MINTP", Priority.MINISTERE,
        ("travaux publics", "route", "pont", "infrastructure", "chantier"),
        ("MINTP Cameroun travaux publics",),
    ),
    OfficialSource(
        "minee.gov.cm", "Ministère de l'Eau et de l'Énergie", "MINEE", Priority.MINISTERE,
        ("eau", "energie", "electricite", "barrage", "forage", "adduction"),
        ("MINEE Cameroun eau énergie",),
    ),
    OfficialSource(
        "mint.gov.cm", "Ministère des Transports", "MINT", Priority.MINISTERE,
        ("transport", "permis de conduire", "vehicule", "immatriculation vehicule",
         "carte grise", "circulation", "aerien", "maritime"),
        ("MINT Cameroun ministère des transports",),
    ),
    OfficialSource(
        "minpostel.gov.cm", "Ministère des Postes et Télécommunications", "MINPOSTEL",
        Priority.MINISTERE,
        ("telecommunication", "poste", "internet", "numerique", "telephonie"),
        ("MINPOSTEL Cameroun postes télécommunications",),
    ),
    OfficialSource(
        "mincom.gov.cm", "Ministère de la Communication", "MINCOM", Priority.MINISTERE,
        ("communication", "presse", "media", "information gouvernementale"),
        ("MINCOM Cameroun ministère de la communication",),
    ),
    OfficialSource(
        "minas.gov.cm", "Ministère des Affaires Sociales", "MINAS", Priority.MINISTERE,
        ("affaires sociales", "handicap", "personne agee", "enfance", "solidarite",
         "aide sociale"),
        ("MINAS Cameroun affaires sociales",),
    ),
    OfficialSource(
        "minproff.gov.cm", "Ministère de la Promotion de la Femme et de la Famille",
        "MINPROFF", Priority.MINISTERE,
        ("femme", "famille", "mariage", "genre", "protection de la famille"),
        ("MINPROFF Cameroun promotion femme famille",),
    ),
    OfficialSource(
        "minjeun.gov.cm", "Ministère de la Jeunesse et de l'Éducation Civique", "MINJEC",
        Priority.MINISTERE,
        ("jeunesse", "education civique", "service civique", "association de jeunes"),
        ("MINJEC Cameroun jeunesse éducation civique",),
    ),
    OfficialSource(
        "minresi.gov.cm", "Ministère de la Recherche Scientifique et de l'Innovation",
        "MINRESI", Priority.MINISTERE,
        ("recherche", "innovation", "science", "brevet", "laboratoire"),
        ("MINRESI Cameroun recherche scientifique innovation",),
    ),
    OfficialSource(
        "mindef.gov.cm", "Ministère de la Défense", "MINDEF", Priority.MINISTERE,
        ("defense", "armee", "militaire", "gendarmerie", "service national"),
        ("MINDEF Cameroun ministère de la défense",),
    ),

    # ── P2 — Administrations et agences spécialisées ──────────────────────────
    OfficialSource(
        "impots.cm", "Direction Générale des Impôts", "DGI", Priority.ADMINISTRATION,
        ("impot", "tva", "declaration fiscale", "contribuable", "niu", "patente",
         "teledeclaration", "fiscalite", "taxe"),
        ("Direction Générale des Impôts Cameroun impots.cm",),
        aliases=("dgi.cm", "teledeclaration-dgi.cm"),  # fiscalis.dgi.cm, mesure.dgi.cm
    ),
    OfficialSource(
        "douanes.cm", "Direction Générale des Douanes", "DOUANES", Priority.ADMINISTRATION,
        ("douane", "importation", "exportation", "dedouanement", "tarif douanier",
         "marchandise"),
        ("Douanes Cameroun direction générale des douanes",),
    ),
    OfficialSource(
        "cnps.cm", "Caisse Nationale de Prévoyance Sociale", "CNPS", Priority.ADMINISTRATION,
        ("cnps", "pension", "retraite", "prestation familiale", "accident du travail",
         "immatriculation sociale", "cotisation"),
        ("CNPS Cameroun caisse nationale de prévoyance sociale",),
    ),
    OfficialSource(
        "dgsn.cm", "Délégation Générale à la Sûreté Nationale", "DGSN", Priority.ADMINISTRATION,
        ("police", "carte nationale d'identite", "cni", "passeport", "surete",
         "commissariat"),
        ("DGSN Cameroun sûreté nationale carte nationale d'identité",),
    ),
    OfficialSource(
        "elecam.cm", "Elections Cameroon", "ELECAM", Priority.ADMINISTRATION,
        ("election", "inscription sur les listes", "vote", "electeur", "scrutin"),
        ("ELECAM Elections Cameroon",),
    ),
    OfficialSource(
        "antic.cm", "Agence Nationale des Technologies de l'Information", "ANTIC",
        Priority.ADMINISTRATION,
        ("cybersecurite", "certificat electronique", "signature electronique",
         "nom de domaine", "numerique"),
        ("ANTIC Cameroun technologies information communication",),
    ),
    OfficialSource(
        "art.cm", "Agence de Régulation des Télécommunications", "ART", Priority.ADMINISTRATION,
        ("telecommunication", "operateur", "regulation telecom", "tarif telephonique"),
        ("ART Cameroun agence régulation télécommunications",),
    ),
    OfficialSource(
        "anor.cm", "Agence des Normes et de la Qualité", "ANOR", Priority.ADMINISTRATION,
        ("norme", "qualite", "certification", "standard", "conformite"),
        ("ANOR Cameroun agence normes qualité",),
    ),
    OfficialSource(
        "armp.cm", "Agence de Régulation des Marchés Publics", "ARMP", Priority.ADMINISTRATION,
        ("marche public", "regulation", "recours", "appel d'offres"),
        ("ARMP Cameroun régulation marchés publics",),
    ),
    OfficialSource(
        "conac.cm", "Commission Nationale Anti-Corruption", "CONAC", Priority.ADMINISTRATION,
        ("corruption", "denonciation", "integrite", "detournement"),
        ("CONAC Cameroun commission nationale anti-corruption",),
    ),
    OfficialSource(
        "cndhl.cm", "Commission des Droits de l'Homme du Cameroun", "CDHC",
        Priority.ADMINISTRATION,
        ("droits de l'homme", "liberte", "discrimination", "plainte"),
        ("Commission droits de l'homme Cameroun",),
    ),
    OfficialSource(
        "ins-cameroun.org", "Institut National de la Statistique", "INS",
        Priority.ADMINISTRATION,
        ("statistique", "recensement", "indicateur", "population", "donnees"),
        ("Institut National de la Statistique Cameroun",),
    ),
    OfficialSource(
        "api.gov.cm", "Agence de Promotion des Investissements", "API",
        Priority.ADMINISTRATION,
        ("investissement", "incitation", "agrement", "investisseur"),
        ("API Cameroun agence promotion investissements",),
    ),

    # ── P3 — Organismes et entreprises publics ────────────────────────────────
    OfficialSource(
        "camtel.cm", "Cameroon Telecommunications", "CAMTEL", Priority.ORGANISME_PUBLIC,
        ("telephone", "internet", "fibre", "abonnement telecom"),
        ("Camtel Cameroun",),
    ),
    OfficialSource(
        "eneo.cm", "Energy of Cameroon", "ENEO", Priority.ORGANISME_PUBLIC,
        ("electricite", "facture electricite", "abonnement electrique", "compteur",
         "delestage"),
        ("Eneo Cameroun électricité",),
    ),
    OfficialSource(
        "caa.cm", "Caisse Autonome d'Amortissement", "CAA", Priority.ORGANISME_PUBLIC,
        ("dette publique", "amortissement", "emprunt"),
        ("Caisse Autonome d'Amortissement Cameroun",),
    ),
    OfficialSource(
        "cnc.cm", "Conseil National de la Communication", "CNC", Priority.ORGANISME_PUBLIC,
        ("regulation media", "deontologie", "presse"),
        ("Conseil National de la Communication Cameroun",),
    ),

    # ── P4 — Organisations régionales ─────────────────────────────────────────
    OfficialSource(
        "ohada.org", "Organisation pour l'Harmonisation en Afrique du Droit des Affaires",
        "OHADA", Priority.INTERNATIONAL,
        ("ohada", "acte uniforme", "societe commerciale", "sarl", "sa", "gie",
         "droit des affaires", "sûrete", "procedure collective", "arbitrage"),
        ("OHADA acte uniforme droit des affaires",),
    ),
    OfficialSource(
        "beac.int", "Banque des États de l'Afrique Centrale", "BEAC", Priority.INTERNATIONAL,
        ("banque centrale", "franc cfa", "change", "monnaie", "taux directeur",
         "reglementation bancaire"),
        ("BEAC banque des états de l'Afrique centrale",),
    ),
    OfficialSource(
        "cemac.int", "Communauté Économique et Monétaire de l'Afrique Centrale", "CEMAC",
        Priority.INTERNATIONAL,
        ("cemac", "integration regionale", "tarif exterieur commun", "libre circulation"),
        ("CEMAC communauté économique monétaire Afrique centrale",),
    ),
    OfficialSource(
        "cameroun.eregulations.org", "eRegulations Cameroun", "eRegulations",
        Priority.INTERNATIONAL,
        ("creation d'entreprise", "creer une entreprise", "entreprise", "societe",
         "formalite", "procedure entreprise", "guichet unique", "cout", "delai",
         "registre du commerce", "immatriculer", "statuts"),
        ("eRegulations Cameroun création entreprise procédure",),
    ),

    # ── P5 — Presse publique (actualité, jamais valeur normative) ─────────────
    OfficialSource(
        "cameroon-tribune.cm", "Cameroon Tribune", "CT", Priority.PRESSE_PUBLIQUE,
        ("actualite", "communique", "nomination", "evenement"),
        ("Cameroon Tribune",),
    ),
    OfficialSource(
        "crtv.cm", "Cameroon Radio Television", "CRTV", Priority.PRESSE_PUBLIQUE,
        ("actualite", "information", "reportage"),
        ("CRTV Cameroun",),
    ),
)


# ── Index et normalisation ────────────────────────────────────────────────────

_ACCENT_RE = re.compile(r"[̀-ͯ]")

# Un sujet court doit correspondre à un mot entier : cherché en sous-chaîne,
# « sa » (société anonyme) matche « sans », « santé », « passage »… et route
# n'importe quelle question vers l'OHADA. Un sujet plus long peut en revanche
# correspondre à un début de mot, pour que « vaccin » couvre « vacciner ».
_WHOLE_WORD_MAX_LEN = 4


def _normalize(text: str) -> str:
    """Minuscules sans accents, pour comparer question et mots-clés de sujet."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return _ACCENT_RE.sub("", decomposed)


def _host_of(url: str) -> str:
    """Extrait le nom d'hôte d'une URL, sans le préfixe www."""
    match = re.match(r"https?://([^/]+)", url.strip(), re.IGNORECASE)
    host = (match.group(1) if match else url).lower()
    return host[4:] if host.startswith("www.") else host


def find_source(url: str) -> Optional[OfficialSource]:
    """
    Retourne la source officielle correspondant à une URL, si elle est inscrite.

    Le rattachement se fait sur le domaine ou l'un de ses sous-domaines :
    `ebulletin.minfi.cm` relève bien du MINFI.
    """
    host = _host_of(url)
    for source in SOURCES:
        for domain in (source.domain, *source.aliases):
            if host == domain or host.endswith("." + domain):
                return source
    return None


def is_official(url: str) -> bool:
    """
    Vrai si l'URL appartient à une source inscrite.

    Indépendant de la disponibilité : c'est l'origine du document qui fonde la
    confiance, pas l'état du serveur au moment de la question.
    """
    return find_source(url) is not None


def _topic_matches(topic: str, normalized_query: str) -> bool:
    """Vrai si le sujet apparaît comme mot (ou début de mot) dans la question."""
    needle = _normalize(topic)
    boundary = r"\b" if len(needle) <= _WHOLE_WORD_MAX_LEN else ""
    pattern = r"\b" + re.escape(needle) + boundary
    return re.search(pattern, normalized_query) is not None


@dataclass
class Routing:
    """Institutions retenues pour une question, et pourquoi."""
    sources: list[OfficialSource] = field(default_factory=list)
    matched_topics: list[str] = field(default_factory=list)

    @property
    def is_specific(self) -> bool:
        """Vrai si la question a pu être rattachée à des compétences précises."""
        return bool(self.matched_topics)


_PROCEDURAL_HINTS = re.compile(
    r"\b(comment|demarche|procedure|obtenir|creer|faire|deposer|demander|"
    r"inscrire|piece[s]?\s+a\s+fournir|formulaire|dossier|guichet|cout|delai|tarif)",
    re.IGNORECASE,
)


def route_question(query: str, max_sources: int = 4) -> Routing:
    """
    Détermine quelles institutions sont compétentes pour une question.

    Le score d'une source est le nombre de ses sujets présents dans la question,
    départagé par sa priorité. Sans correspondance, on retombe sur les sources
    généralistes : le portail des services publics et les institutions suprêmes,
    qui couvrent l'essentiel des démarches.
    """
    normalized = _normalize(query)

    scored: list[tuple[int, int, OfficialSource, list[str]]] = []
    for source in SOURCES:
        # Une source injoignable reste routable : le moteur peut en servir une
        # page indexée, et l'écarter priverait la question de son autorité
        # compétente.
        hits = [topic for topic in source.topics if _topic_matches(topic, normalized)]
        if hits:
            scored.append((len(hits), -int(source.priority), source, hits))

    if not scored:
        generalists = [
            s for s in SOURCES if s.priority == Priority.INSTITUTION_SUPREME
        ][:max_sources]
        return Routing(sources=generalists, matched_topics=[])

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = scored[:max_sources]
    topics: list[str] = []
    for _, _, _, hits in selected:
        topics.extend(hits)

    sources = [source for _, _, source, _ in selected]

    # Le portail des services publics décrit les pièces, coûts et guichets de
    # toute démarche, quel que soit le ministère compétent : sur une question
    # procédurale il complète utilement l'autorité sectorielle.
    if _PROCEDURAL_HINTS.search(_normalize(query)):
        portal = next((s for s in SOURCES if s.acronym == "ServicePublic"), None)
        if portal is not None and portal not in sources:
            sources = ([portal] + sources)[:max_sources]

    return Routing(
        sources=sources,
        matched_topics=list(dict.fromkeys(topics)),
    )


def registry_stats() -> dict[str, int]:
    """Volumétrie du registre, pour la page Santé et les journaux."""
    stats = {"total": len(SOURCES), "reachable": sum(1 for s in SOURCES if s.reachable)}
    for priority in Priority:
        stats[priority.name.lower()] = sum(1 for s in SOURCES if s.priority == priority)
    return stats
