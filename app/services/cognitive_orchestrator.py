"""
GOV-AI 2.0 — Orchestrateur Cognitif (Algorithme 1, §4.3.1).
Cycle : PERCEVOIR → COMPRENDRE → DÉLIBÉRER → AGIR → VÉRIFIER → ADAPTER

Chaque étape correspond à un nœud dans le graphe d'état LangGraph.
En l'absence de LangGraph, un pipeline séquentiel équivalent est exécuté.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.domain import (
    ConfidenceLevel,
    IntentType,
    Language,
    SafetyFlag,
    UserProfile,
)
from app.models.schemas import QueryRequest, QueryResponse, RetrievedChunk

logger = get_logger(__name__)

# ── Mots-clés pour la classification d'intention ──────────────────────────────

_INTENT_KEYWORDS_FR: dict[IntentType, list[str]] = {
    IntentType.PROCEDURAL: [
        "comment", "procédure", "démarche", "étapes", "formulaire",
        "obtenir", "demander", "faire", "créer", "enregistrer", "immatriculer",
    ],
    IntentType.NORMATIVE: [
        "loi", "article", "décret", "arrêté", "code", "texte", "réglementation",
        "obligation", "interdit", "autorisé", "sanctionné", "peine", "amende",
    ],
    IntentType.COMPARATIVE: [
        "différence", "comparer", "vs", "versus", "droit civil", "common law",
        "anglophone", "francophone", "ohada", "national", "régional",
    ],
    IntentType.DOCUMENT_REQUEST: [
        "document", "certificat", "acte", "copie", "attestation",
        "extrait", "bulletin", "titre", "carte", "passeport",
    ],
    IntentType.FACTUAL: [
        "qu'est-ce", "définition", "signifie", "quel", "qui", "où", "quand",
        "combien", "délai", "durée", "montant", "frais", "coût",
    ],
}

_INTENT_KEYWORDS_EN: dict[IntentType, list[str]] = {
    IntentType.PROCEDURAL: [
        "how", "procedure", "steps", "process", "obtain", "apply", "register",
        "create", "file", "submit", "request",
    ],
    IntentType.NORMATIVE: [
        "law", "article", "decree", "regulation", "code", "rule", "prohibited",
        "allowed", "obligation", "penalty", "fine", "sanction",
    ],
    IntentType.COMPARATIVE: [
        "difference", "compare", "vs", "versus", "civil law", "common law",
        "anglophone", "francophone", "ohada",
    ],
    IntentType.DOCUMENT_REQUEST: [
        "document", "certificate", "copy", "extract", "attestation",
        "birth", "marriage", "passport", "card",
    ],
    IntentType.FACTUAL: [
        "what", "who", "where", "when", "how much", "how long",
        "cost", "fee", "deadline", "definition",
    ],
}

# ── Bavardage : salutations, remerciements, questions sur l'assistant ─────────
# Doit couvrir la requête ENTIÈRE : « bonjour » est du bavardage, « bonjour, quelle
# est la durée de la garde à vue ? » est une vraie question.
_CHITCHAT_PATTERN = re.compile(
    r"^\W*(?:"
    r"(?:bonjour|bonsoir|salut|coucou|hello|hi|hey|good\s+(?:morning|evening))|"
    r"(?:merci|thanks?|thank\s+you|ok|d'accord)|"
    r"(?:(?:qui|que|what)\s+(?:es-tu|etes-vous|êtes-vous|are\s+you)"
    r"|tu\s+(?:es|fais)\s+quoi|comment\s+(?:ca|ça)\s+va|how\s+are\s+you"
    r"|(?:que|quoi)\s+(?:peux-tu|pouvez-vous)\s+faire|what\s+can\s+you\s+do)"
    r")\W*$",
    re.IGNORECASE,
)

# ── Requêtes agrégatives : vue d'ensemble d'un document entier ────────────────
# Le retrieval top-k ne peut pas y répondre — cinq extraits ne décrivent pas un
# code de plusieurs centaines d'articles.
_AGGREGATIVE_PATTERN = re.compile(
    r"\b("
    r"resum|résum|synthes|synthès|vue\s+d.ensemble|panorama|apercu|aperçu|"
    r"sommaire|table\s+des\s+mati|plan\s+(?:du|de\s+la|des)|structure\s+(?:du|de\s+la|des)|"
    r"que\s+contient|de\s+quoi\s+(?:parle|traite)|qu.est-ce\s+que\s+contient|"
    r"presente[rz]?\s+(?:moi\s+)?(?:le|la|les)|"
    r"(?:liste|enumere|énumère|donne)\s+(?:moi\s+)?(?:tous|toutes|l.ensemble)|"
    r"summar|overview|outline|table\s+of\s+contents|what\s+does\s+.{0,20}contain|"
    r"list\s+all|describe\s+the\s+(?:whole|entire)"
    r")",
    re.IGNORECASE,
)

# ── Mots-clés hors-domaine ────────────────────────────────────────────────────

_OUT_OF_SCOPE_PATTERNS = re.compile(
    r"\b(météo|sport|football|recette|cuisine|film|musique|jeu|"
    r"weather|sport|recipe|movie|game|crypto|bitcoin)\b",
    re.IGNORECASE,
)

# ── Injection prompts ─────────────────────────────────────────────────────────

_INJECTION_PATTERNS = re.compile(
    r"(ignore (previous|all) instructions?|"
    r"forget (your|all) (rules?|instructions?)|"
    r"you are now|act as|jailbreak|"
    r"oublie tes instructions|tu es maintenant|fais semblant d'être)",
    re.IGNORECASE,
)


# ── État de l'orchestrateur ───────────────────────────────────────────────────

@dataclass
class OrchestratorState:
    """
    État partagé entre tous les nœuds du graphe cognitif.
    Équivalent au StateGraph LangGraph — compatible avec et sans LangGraph.
    """
    # Entrée
    request: QueryRequest
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # PERCEVOIR
    language: Language = Language.UNKNOWN
    intent: IntentType = IntentType.UNKNOWN
    entities: list[str] = field(default_factory=list)
    safety_flags: list[SafetyFlag] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # COMPRENDRE
    profile: UserProfile = UserProfile.CITIZEN
    session_context: str = ""

    # DÉLIBÉRER
    action_plan: list[str] = field(default_factory=list)

    # AGIR
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    retrieval_score: float = 0.0
    kg_context_str: str = ""
    # Même contenu que `kg_context_str`, sous forme d'objets : l'un part vers le
    # modèle, l'autre vers l'interface. Ce que le graphe a dit au modèle doit
    # être exactement ce que l'utilisateur peut inspecter.
    kg_evidence: list = field(default_factory=list)
    # Ossature du corpus pour les requêtes agrégatives (volume, plan, plage
    # d'articles) — des faits indexés, distincts des extraits citables.
    survey_outline: str = ""
    # Pages attestées par l'index, par document : une référence à l'ossature
    # est exacte sans être adossée à un extrait.
    survey_index_pages: dict[str, set[int]] = field(default_factory=dict)
    # Sources web officielles : bloc de contexte distinct du corpus et du graphe,
    # car leur statut probatoire diffère — une page ministérielle présente le
    # droit, elle ne l'est pas.
    web_context_str: str = ""
    web_results: list[dict[str, Any]] = field(default_factory=list)
    # Passages issus des pièces jointes de CETTE conversation. Comptés à part
    # pour pouvoir les annoncer comme tels : ils ne font pas partie du corpus.
    session_chunks: int = 0

    # GÉNÉRER
    response: Optional[QueryResponse] = None

    # VÉRIFIER
    confidence_score: float = 0.0
    escalation_needed: bool = False

    # Métriques
    start_time: float = field(default_factory=time.perf_counter)
    step_latencies: dict[str, float] = field(default_factory=dict)


# ── Orchestrateur principal ───────────────────────────────────────────────────

class CognitiveOrchestrator:
    """
    Orchestre le pipeline cognitif GOV-AI 2.0.

    Implémente l'Algorithme 1 du mémoire :
      PERCEVOIR → COMPRENDRE → DÉLIBÉRER → AGIR → GÉNÉRER → VÉRIFIER → ADAPTER

    Utilise LangGraph si disponible, sinon pipeline séquentiel équivalent.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._retrieval_service: Any = None
        self._generation_service: Any = None
        self._reranking_service: Any = None
        self._kg_service: Any = None

    # ── Services lazy ─────────────────────────────────────────────────────────

    def _get_reranking_service(self) -> Any:
        if self._reranking_service is None:
            from app.services.reranking.reranking_service import RerankingService
            self._reranking_service = RerankingService()
        return self._reranking_service

    def _get_retrieval_service(self) -> Any:
        if self._retrieval_service is None:
            from app.services.retrieval.hybrid_retrieval_service import HybridRetrievalService
            from app.services.embedding import get_embedding_service
            svc = HybridRetrievalService()
            svc.set_embedding_service(get_embedding_service())
            svc.set_reranking_service(self._get_reranking_service())
            self._retrieval_service = svc
        return self._retrieval_service

    def _get_generation_service(self) -> Any:
        if self._generation_service is None:
            from app.services.generation.llm_answer_service import LLMAnswerService
            self._generation_service = LLMAnswerService()
        return self._generation_service

    def _get_kg_service(self) -> Any:
        if self._kg_service is None:
            try:
                from app.services.knowledge_graph.kg_service import KnowledgeGraphService
                self._kg_service = KnowledgeGraphService()
            except Exception as exc:
                logger.warning("kg_service_unavailable", error=str(exc))
        return self._kg_service

    # ── Point d'entrée principal ──────────────────────────────────────────────

    async def process(self, request: QueryRequest) -> QueryResponse:
        """
        Traite une requête complète via le cycle cognitif (Algo 1).

        Args:
            request: QueryRequest validé (query, language, profile, session_id)

        Returns:
            QueryResponse avec réponse, citations, flags, métriques
        """
        state = OrchestratorState(request=request)

        logger.info(
            "orchestrator_start",
            trace_id=state.trace_id,
            query_preview=request.query[:80],
            session_id=str(request.session_id),
        )

        try:
            # Cycle cognitif séquentiel (MVP sans LangGraph)
            state = await self._percevoir(state)
            state = await self._comprendre(state)
            state = await self._deliberer(state)
            state = await self._agir(state)
            state = await self._generer(state)
            state = await self._verifier(state)
            state = await self._adapter(state)
        except Exception as exc:
            logger.error(
                "orchestrator_pipeline_error",
                trace_id=state.trace_id,
                error=str(exc),
                exc_info=True,
            )
            return self._build_error_response(state, exc)

        # Injecter session_id, intent et plan dans la réponse finale
        if state.response is not None:
            state.response = state.response.model_copy(update={
                "session_id": state.request.session_id,
                "intent_detected": state.intent,
                "juridical_system_detected": state.request.juridical_system,
                "action_plan": state.action_plan,
                "graph_evidence": state.kg_evidence,
            })

        total_ms = (time.perf_counter() - state.start_time) * 1000
        logger.info(
            "orchestrator_complete",
            trace_id=state.trace_id,
            total_ms=round(total_ms, 2),
            confidence=round(state.confidence_score, 3),
            escalation=state.escalation_needed,
            citations=len(state.response.citations) if state.response else 0,
        )

        return state.response  # type: ignore[return-value]

    async def process_stream(self, request: QueryRequest):
        """
        Même cycle cognitif que `process`, mais la génération est diffusée au fil
        de l'eau. Rend des événements structurés, dans cet ordre :

          {"type": "meta",  ...}   après le retrieval : intention, plan d'action,
                                   nombre d'extraits, identifiant de session ;
          {"type": "token", "v": …} chaque fragment de texte produit ;
          {"type": "done",  ...}   citations résolues, drapeaux, avertissements,
                                   latence ;
          {"type": "error", ...}   en cas d'échec, à la place de "done".

        Les citations ne peuvent être résolues qu'une fois le texte complet : ce
        sont les références réellement écrites qui déterminent les sources.
        L'ancien chemin streaming court-circuitait l'orchestrateur — ni routage,
        ni reranking, ni citations, ni persistance de session.
        """
        state = OrchestratorState(request=request)
        logger.info(
            "orchestrator_stream_start",
            trace_id=state.trace_id,
            query_preview=request.query[:80],
        )

        # Étapes annoncées au fil de l'eau : sur cette machine, la récupération
        # et le reranking prennent près de deux minutes avant le premier token.
        # Sans ces jalons, l'interface reste muette et paraît figée.
        try:
            yield {"type": "stage", "stage": "analyzing", "label": "Analyse de la question"}
            state = await self._percevoir(state)
            state = await self._comprendre(state)
            state = await self._deliberer(state)

            if "STRUCTURAL_SURVEY" in state.action_plan:
                yield {
                    "type": "stage",
                    "stage": "surveying",
                    "label": "Lecture de la structure du document",
                }
            elif "HYBRID_RAG" in state.action_plan:
                yield {
                    "type": "stage",
                    "stage": "retrieving",
                    "label": "Recherche dans le corpus et reclassement des extraits",
                }

            if "WEB_SEARCH_FORCED" in state.action_plan:
                yield {
                    "type": "stage",
                    "stage": "websearch",
                    "label": "Consultation des sources officielles en ligne",
                }

            state = await self._agir(state)
        except Exception as exc:
            logger.error(
                "orchestrator_stream_prepare_failed",
                trace_id=state.trace_id, error=str(exc), exc_info=True,
            )
            yield {"type": "error", "message": f"Erreur de préparation : {exc}"}
            return

        yield {
            "type": "meta",
            "trace_id": state.trace_id,
            "session_id": str(request.session_id) if request.session_id else None,
            "intent": state.intent.value,
            "action_plan": state.action_plan,
            "language": state.language.value,
            "profile": state.profile.value,
            "chunks_retrieved": len(state.retrieved_chunks),
        }

        # Refus et bavardage : la réponse est déterministe, aucun appel au modèle.
        direct_plans = ("REFUSE_INJECTION", "REFUSE_OUT_OF_SCOPE", "ANSWER_CHITCHAT")
        if any(plan in state.action_plan for plan in direct_plans):
            state = await self._generer(state)
            answer = state.response.answer if state.response else ""
            yield {"type": "token", "v": answer}
            yield {
                "type": "done",
                "answer": answer,
                "citations": [],
                "safety_flags": [f.value for f in (state.response.safety_flags if state.response else [])],
                "warnings": state.response.warnings if state.response else [],
                "model_used": state.response.model_used if state.response else None,
                "latency_ms": round((time.perf_counter() - state.start_time) * 1000, 2),
            }
            return

        if not state.retrieved_chunks:
            generation_service = self._get_generation_service()
            insufficient = generation_service._build_insufficient_response(
                request.query, state.language, state.retrieval_score, "no_chunks"
            )
            yield {"type": "token", "v": insufficient.answer}
            yield {
                "type": "done",
                "answer": insufficient.answer,
                "citations": [],
                "safety_flags": [f.value for f in insufficient.safety_flags],
                "warnings": insufficient.warnings,
                "model_used": insufficient.model_used,
                "latency_ms": round((time.perf_counter() - state.start_time) * 1000, 2),
            }
            return

        # Génération diffusée
        generation_service = self._get_generation_service()
        combined_context = self._combine_context(state)

        yield {"type": "stage", "stage": "generating", "label": "Rédaction de la réponse"}

        pieces: list[str] = []
        try:
            async for token in generation_service.generate_stream(
                query=request.query,
                retrieved_chunks=state.retrieved_chunks,
                language=state.language,
                profile=state.profile,
                session_context=combined_context,
                survey_outline=state.survey_outline,
                web_context=state.web_context_str,
            ):
                pieces.append(token)
                yield {"type": "token", "v": token}
        except Exception as exc:
            logger.error(
                "orchestrator_stream_generation_failed",
                trace_id=state.trace_id, error=str(exc), exc_info=True,
            )
            yield {"type": "error", "message": f"Erreur de génération : {exc}"}
            return

        answer = "".join(pieces)

        # Résolution des citations sur le texte complet, puis vérification et
        # persistance : le fil de conversation doit être rejouable à l'identique.
        citations, unresolved = generation_service._extract_citations(
            answer, state.retrieved_chunks, state.survey_index_pages or None
        )
        safety_flags: list[SafetyFlag] = []
        warnings: list[str] = list(state.warnings)

        if not citations:
            safety_flags.append(SafetyFlag.UNSUPPORTED_CLAIMS)
            warnings.append(
                "Attention : la réponse ne contient pas de citations vérifiables. "
                "Vérifiez les sources manuellement."
            )
        if unresolved:
            if SafetyFlag.UNSUPPORTED_CLAIMS not in safety_flags:
                safety_flags.append(SafetyFlag.UNSUPPORTED_CLAIMS)
            distinct = list(dict.fromkeys(unresolved))[:5]
            warnings.append(
                "Références citées introuvables dans le contexte fourni : " + " ; ".join(distinct)
            )

        latency_ms = round((time.perf_counter() - state.start_time) * 1000, 2)
        from app.models.schemas import WebSource

        state.response = QueryResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=state.retrieved_chunks,
            web_sources=[WebSource(**r) for r in state.web_results],
            graph_evidence=state.kg_evidence,
            uncertainty_score=state.retrieval_score,
            safety_flags=safety_flags,
            warnings=warnings,
            language_detected=state.language,
            latency_ms=latency_ms,
            model_used=self.settings.llm_model,
        )

        try:
            state = await self._verifier(state)
            state = await self._adapter(state)
        except Exception as exc:
            logger.warning("orchestrator_stream_postprocess_failed", error=str(exc))

        response = state.response
        yield {
            "type": "done",
            "answer": answer,
            "citations": [c.model_dump(mode="json") for c in response.citations],
            "web_sources": [w.model_dump(mode="json") for w in response.web_sources],
            "graph_evidence": [g.model_dump(mode="json") for g in response.graph_evidence],
            "safety_flags": [f.value for f in response.safety_flags],
            "warnings": response.warnings,
            "confidence_level": response.confidence_level.value,
            "uncertainty_score": response.uncertainty_score,
            "intent": state.intent.value,
            "model_used": response.model_used,
            "latency_ms": latency_ms,
        }

        logger.info(
            "orchestrator_stream_complete",
            trace_id=state.trace_id,
            total_ms=latency_ms,
            citations=len(response.citations),
        )

    # ── Nœud 1 : PERCEVOIR ────────────────────────────────────────────────────

    async def _percevoir(self, state: OrchestratorState) -> OrchestratorState:
        """
        PERCEVOIR : normaliser la requête, détecter la langue, classifier l'intention.

        Sous-étapes (Algo 1) :
          - détecterLangue(q) → L
          - extraireEntités(q) → E
          - classifierIntention(q, L) → I
          - vérifierSécurité(q) → flags
        """
        t0 = time.perf_counter()
        query = state.request.query

        # ── Sécurité : injection de prompts ──────────────────────────────────
        if _INJECTION_PATTERNS.search(query):
            state.safety_flags.append(SafetyFlag.PROMPT_INJECTION_ATTEMPT)
            state.warnings.append(
                "Tentative d'injection de prompts détectée. Requête neutralisée."
            )
            logger.warning(
                "prompt_injection_detected",
                trace_id=state.trace_id,
                query_preview=query[:80],
            )

        # ── Détection de langue ───────────────────────────────────────────────
        if state.request.language and state.request.language != Language.UNKNOWN:
            state.language = state.request.language
        else:
            state.language = self._detect_language(query)

        # ── Extraction d'entités légères ──────────────────────────────────────
        state.entities = self._extract_entities(query)

        # ── Classification d'intention ────────────────────────────────────────
        state.intent = self._classify_intent(query, state.language)

        # ── Hors-domaine ──────────────────────────────────────────────────────
        if state.intent == IntentType.OUT_OF_SCOPE:
            state.safety_flags.append(SafetyFlag.OUT_OF_CORPUS)
            state.warnings.append(
                "Cette requête semble hors du domaine de l'administration camerounaise."
            )

        state.step_latencies["percevoir"] = (time.perf_counter() - t0) * 1000
        logger.debug(
            "percevoir_done",
            trace_id=state.trace_id,
            language=state.language,
            intent=state.intent,
            entities=state.entities,
        )
        return state

    # ── Nœud 2 : COMPRENDRE ───────────────────────────────────────────────────

    async def _comprendre(self, state: OrchestratorState) -> OrchestratorState:
        """
        COMPRENDRE : charger le profil utilisateur, résumer le contexte de session.

        Sous-étapes (Algo 1) :
          - chargerProfil(session_id) → profil P
          - résumerSession(historique) → contexte S  [Sprint 3 : depuis PostgreSQL]
        """
        t0 = time.perf_counter()

        # Profil utilisateur
        state.profile = state.request.profile or UserProfile.CITIZEN

        # Contexte de session : d'abord depuis PostgreSQL, sinon depuis la requête
        if state.request.session_id:
            db_context = await self._load_session_history(state.request.session_id)
            if db_context:
                state.session_context = db_context
            elif state.request.session_context:
                state.session_context = state.request.session_context
        elif state.request.session_context:
            state.session_context = state.request.session_context

        state.step_latencies["comprendre"] = (time.perf_counter() - t0) * 1000
        logger.debug(
            "comprendre_done",
            trace_id=state.trace_id,
            profile=state.profile,
            has_session_context=bool(state.session_context),
        )
        return state

    async def _load_session_history(self, session_id: str) -> str:
        """Charge les N derniers échanges d'une session depuis PostgreSQL (Sprint 3)."""
        try:
            from app.storage.postgres_client import get_db_session
            from app.models.db_models import QueryLog
            from sqlalchemy import select

            async with get_db_session() as db:
                result = await db.execute(
                    select(QueryLog)
                    .where(QueryLog.session_id == session_id)
                    .order_by(QueryLog.created_at.desc())
                    .limit(self.settings.session_history_max_turns)
                )
                logs = result.scalars().all()

            if not logs:
                return ""

            turns: list[str] = []
            for log in reversed(logs):
                answer_preview = ""
                if log.response_json:
                    answer_preview = log.response_json.get("answer_preview", "")
                turns.append(f"Q: {log.query}\nR: {answer_preview}")

            return "\n\n".join(turns)
        except Exception as exc:
            logger.warning("session_history_load_failed", error=str(exc))
            return ""

    # ── Nœud 3 : DÉLIBÉRER ───────────────────────────────────────────────────

    async def _deliberer(self, state: OrchestratorState) -> OrchestratorState:
        """
        DÉLIBÉRER : choisir les actions à exécuter selon l'intention.

        Plan d'action (Sprint 1) :
          - Toujours : RAG hybride (dense + BM25 + reranking)
          - OUT_OF_SCOPE → réponse refus directe
          - INJECTION → réponse refus directe
        """
        t0 = time.perf_counter()

        if SafetyFlag.PROMPT_INJECTION_ATTEMPT in state.safety_flags:
            state.action_plan = ["REFUSE_INJECTION"]
        elif state.intent == IntentType.OUT_OF_SCOPE:
            state.action_plan = ["REFUSE_OUT_OF_SCOPE"]
        elif state.intent == IntentType.CHITCHAT:
            # Aucune recherche : elle ne produirait que du hors-sujet cité.
            state.action_plan = ["ANSWER_CHITCHAT"]
        elif state.intent == IntentType.AGGREGATIVE:
            # Parcours structurel du corpus au lieu du top-k sémantique.
            state.action_plan = ["STRUCTURAL_SURVEY"]
        else:
            state.action_plan = ["HYBRID_RAG"]
            # KG activé pour les intentions normatives et comparatives, et pour
            # les questions de compétence — quelle qu'en soit l'intention
            # classée. « Quelle autorité peut décerner un mandat de détention
            # provisoire » est étiquetée `factual_query`, et n'atteignait donc
            # jamais le graphe, alors que c'est le cas qu'il traite le mieux :
            # l'organe compétent est porté par une arête, pas par une
            # proximité sémantique. Le coût est d'une requête Neo4j.
            if (
                state.intent in (IntentType.NORMATIVE, IntentType.COMPARATIVE)
                or self._concerns_an_institution(state.request.query)
            ):
                state.action_plan.append("KNOWLEDGE_GRAPH")
            # Recherche web : la requête l'emporte sur le réglage global, ce qui
            # permet à l'interface de l'exposer comme un bouton par message.
            # Forcée → toujours ; interdite → jamais ; non précisée → réglage
            # global, avec déclenchement conditionnel si la confiance est basse.
            requested = state.request.web_search
            if requested is True:
                state.action_plan.append("WEB_SEARCH_FORCED")
            elif requested is None and self.settings.web_search_enabled:
                state.action_plan.append("WEB_SEARCH_IF_LOW_CONFIDENCE")

        state.step_latencies["deliberer"] = (time.perf_counter() - t0) * 1000
        logger.debug(
            "deliberer_done",
            trace_id=state.trace_id,
            action_plan=state.action_plan,
        )
        return state

    # ── Nœud 4 : AGIR ────────────────────────────────────────────────────────

    async def _agir(self, state: OrchestratorState) -> OrchestratorState:
        """
        AGIR : exécuter le plan (retrieval RAG, graphe, web, mémoire).
        """
        t0 = time.perf_counter()

        if (
            "REFUSE_INJECTION" in state.action_plan
            or "REFUSE_OUT_OF_SCOPE" in state.action_plan
            or "ANSWER_CHITCHAT" in state.action_plan
        ):
            # Pas de retrieval nécessaire
            state.retrieved_chunks = []
            # Le score de retrieval alimente l'escalade dans VÉRIFIER. Pour un
            # bavardage il n'y a pas eu de recherche : un score nul y ferait
            # conclure à une confiance faible et conseillerait de consulter un
            # professionnel — en réponse à « bonjour ».
            state.retrieval_score = (
                1.0 if "ANSWER_CHITCHAT" in state.action_plan else 0.0
            )
            state.step_latencies["agir"] = (time.perf_counter() - t0) * 1000
            return state

        if "STRUCTURAL_SURVEY" in state.action_plan:
            state = await self._execute_structural_survey(state)
            state.step_latencies["agir"] = (time.perf_counter() - t0) * 1000
            return state

        if "HYBRID_RAG" in state.action_plan:
            state = await self._execute_hybrid_rag(state)

        if "KNOWLEDGE_GRAPH" in state.action_plan:
            state = await self._execute_knowledge_graph(state)

        # Web search : forcée par la requête, ou déclenchée si la confiance
        # documentaire est trop basse après retrieval (Sprint 3).
        if "WEB_SEARCH_FORCED" in state.action_plan or (
            "WEB_SEARCH_IF_LOW_CONFIDENCE" in state.action_plan
            and state.retrieval_score < self.settings.escalation_threshold
        ):
            state = await self._execute_web_search(state)

        state.step_latencies["agir"] = (time.perf_counter() - t0) * 1000
        logger.debug(
            "agir_done",
            trace_id=state.trace_id,
            chunks_retrieved=len(state.retrieved_chunks),
            retrieval_score=round(state.retrieval_score, 4),
        )
        return state

    async def _execute_hybrid_rag(self, state: OrchestratorState) -> OrchestratorState:
        """Exécute le retrieval hybride (Algo 2 : dense + BM25 + RRF + reranking)."""
        try:
            retrieval_service = self._get_retrieval_service()
            chunks = await retrieval_service.retrieve(
                query=state.request.query,
                language=state.language,
                top_k=self.settings.retrieval_final_top_k,
                filters=self._build_retrieval_filters(state),
            )
            # Les pièces jointes de la conversation passent devant : quand
            # l'utilisateur attache un document et pose une question, c'est de
            # CE document qu'il parle, pas du corpus général.
            session_chunks = await self._retrieve_session_documents(state)
            if session_chunks:
                budget = self.settings.retrieval_final_top_k
                chunks = (session_chunks + chunks)[:max(budget, len(session_chunks))]
                state.session_chunks = len(session_chunks)

            state.retrieved_chunks = chunks

            # Score de confiance = similarité cosinus du meilleur chunk
            # (dense_score ∈ [0,1], calibré pour les seuils τ_conf/τ_esc)
            if chunks:
                best = chunks[0]
                state.retrieval_score = best.dense_score if best.dense_score is not None else best.final_score
            else:
                state.retrieval_score = 0.0
        except Exception as exc:
            logger.error(
                "hybrid_rag_failed",
                trace_id=state.trace_id,
                error=str(exc),
            )
            state.retrieved_chunks = []
            state.retrieval_score = 0.0
            state.warnings.append(f"Erreur de retrieval : {exc}")

        return state

    def _combine_context(self, state: OrchestratorState) -> str:
        """
        Assemble les contextes annexes du prompt : session, graphe, sources web.

        Chaque bloc reste identifiable — le modèle doit pouvoir distinguer un
        rappel de conversation d'une page ministérielle.
        """
        # Le contexte web n'entre PAS ici : il est passé séparément au
        # constructeur de prompt, qui lui donne sa propre section. Fondu dans le
        # contexte de conversation, le modèle l'ignorait.
        blocks = [
            block for block in (state.kg_context_str, state.session_context) if block
        ]
        return "\n\n".join(blocks)

    async def _retrieve_session_documents(self, state: OrchestratorState) -> list[RetrievedChunk]:
        """
        Cherche dans les documents attachés à cette conversation.

        Ils sont stockés hors des index partagés : la recherche se fait donc
        séparément, puis leurs passages sont placés en tête. Un échec ici ne
        doit pas emporter la réponse — le corpus reste disponible.
        """
        session_id = state.request.session_id
        if not session_id:
            return []

        try:
            from app.services.ingestion.session_documents import (
                get_session_document_service,
            )

            return await get_session_document_service().search(
                session_id=str(session_id),
                query=state.request.query,
                top_k=3,
            )
        except Exception as exc:
            logger.warning(
                "session_documents_search_failed",
                trace_id=state.trace_id,
                error=str(exc),
            )
            return []

    async def _execute_structural_survey(self, state: OrchestratorState) -> OrchestratorState:
        """
        Parcours structurel du corpus, pour les requêtes agrégatives.

        Le retrieval top-k rapporte les cinq passages les plus proches de la
        question : cela répond à « quelles sont les conditions de la garde à vue »,
        pas à « résume le Code pénal ». On substitue ici l'ossature réelle du
        document — volume, plan, plage de dispositions — et un échantillon
        d'extraits réparti sur toute sa longueur.
        """
        try:
            from app.services.retrieval.corpus_outline import build_survey

            outline, chunks, index_pages = await build_survey(state.request.query)
            state.survey_outline = outline
            state.survey_index_pages = index_pages
            state.retrieved_chunks = chunks
            # Le score de confiance du top-k n'a pas de sens ici : les extraits ne
            # sont pas sélectionnés par proximité sémantique mais par position.
            # On neutralise l'escalade plutôt que de la déclencher à tort.
            state.retrieval_score = 1.0 if chunks else 0.0

            if not chunks:
                state.warnings.append(
                    "Aucun document dans le corpus : impossible d'en produire une vue d'ensemble."
                )
        except Exception as exc:
            logger.error(
                "structural_survey_failed",
                trace_id=state.trace_id,
                error=str(exc),
                exc_info=True,
            )
            state.survey_outline = ""
            state.survey_index_pages = {}
            state.retrieved_chunks = []
            state.retrieval_score = 0.0
            state.warnings.append(f"Erreur de parcours structurel : {exc}")

        return state

    @staticmethod
    def _concerns_an_institution(query: str) -> bool:
        """
        La question nomme une institution, ou en cherche une.

        Les deux cas mènent au graphe par des chemins opposés : une institution
        nommée est retrouvée par son libellé ; une institution cherchée l'est
        par les articles que le retrieval remonte.
        """
        from app.services.knowledge_graph.institutions import (
            asks_about_authority,
            find_institutions,
        )

        return bool(find_institutions(query)) or asks_about_authority(query)

    @staticmethod
    def _retrieved_article_ids(state: OrchestratorState) -> list[str]:
        """
        Identifiants de graphe des articles retrouvés, sans doublon.

        L'identifiant d'un nœud `Article` est « {doc_id}:{numéro} » ; il est
        donc reconstitué ici depuis `article_ref`, plutôt que porté par le
        passage — le retrieval ignore le graphe, et c'est bien ainsi.

        Les continuations « (suite) » désignent l'article précédent, dont le
        numéro est déjà présent : les retenir n'ajouterait aucun nœud.
        """
        from app.services.knowledge_graph.graph_builder import article_number

        ids: list[str] = []
        seen: set[str] = set()
        for chunk in state.retrieved_chunks[:20]:
            ref = (chunk.metadata or {}).get("article_ref") or ""
            number = article_number(ref)
            if not number or not chunk.doc_id:
                continue
            article_id = f"{chunk.doc_id}:{number}"
            if article_id not in seen:
                seen.add(article_id)
                ids.append(article_id)
        return ids

    async def _execute_knowledge_graph(self, state: OrchestratorState) -> OrchestratorState:
        """Enrichit le contexte via le graphe de connaissances (Algo 3)."""
        try:
            kg_svc = self._get_kg_service()
            if kg_svc is None:
                return state

            # Le graphe s'exécute après le retrieval : les passages retrouvés
            # sont disponibles, et permettent la traversée inverse — remonter
            # des dispositions vers les autorités qu'elles nomment. Sans cela,
            # « quelle autorité peut… » ne déclenchait rien, faute de nommer
            # l'institution que la question cherche précisément.
            article_ids = self._retrieved_article_ids(state)

            ctx = await kg_svc.enrich_context(
                entities=state.entities,
                query=state.request.query,
                language=state.language.value,
                article_ids=article_ids,
            )
            state.kg_context_str = kg_svc.kg_context_to_prompt_str(ctx, state.language.value)
            state.kg_evidence = kg_svc.kg_context_to_evidence(ctx)

            logger.debug(
                "knowledge_graph_done",
                trace_id=state.trace_id,
                articles=len(ctx.articles),
                institutions=len(ctx.institutions),
                seed_articles=len(article_ids),
                divergences=len(ctx.bijuridical_divergences),
            )
        except Exception as exc:
            logger.warning("knowledge_graph_failed", error=str(exc))

        return state

    async def _execute_web_search(self, state: OrchestratorState) -> OrchestratorState:
        """
        Recherche web restreinte aux sources officielles camerounaises.

        La question est d'abord routée vers les institutions compétentes — c'est
        ce qui fait remonter les sites officiels — puis les résultats sont
        filtrés contre la liste blanche. Tout ce qui n'en relève pas est écarté
        plutôt que présenté comme officiel.
        """
        from app.services.websearch.web_search_service import (
            format_for_prompt,
            get_web_search_service,
        )

        outcome = await get_web_search_service().search(state.request.query)
        state.web_results = [r.to_dict() for r in outcome.results]

        if outcome.error:
            state.warnings.append(outcome.error)
            logger.warning("web_search_error", trace_id=state.trace_id, error=outcome.error)
            return state

        if not outcome.found:
            institutions = ", ".join(s.acronym for s in (outcome.routing.sources if outcome.routing else []))
            state.warnings.append(
                "Aucune page officielle trouvée sur le web pour cette question"
                + (f" (institutions consultées : {institutions})." if institutions else ".")
            )
            return state

        state.web_context_str = format_for_prompt(outcome)
        state.action_plan.append("WEB_SEARCH")

        stale = [r for r in outcome.results if not r.source.reachable]
        if stale:
            state.warnings.append(
                "Certaines sources citées étaient injoignables au dernier contrôle : "
                + ", ".join(sorted({r.source.acronym for r in stale}))
                + ". L'information peut ne plus être à jour."
            )

        logger.info(
            "web_search_done",
            trace_id=state.trace_id,
            kept=len(outcome.results),
            discarded=outcome.discarded,
            institutions=[s.acronym for s in (outcome.routing.sources if outcome.routing else [])],
        )
        return state

    def _build_retrieval_filters(self, state: OrchestratorState) -> dict[str, Any]:
        """Construit les filtres Milvus/ES selon le profil et l'intention."""
        filters: dict[str, Any] = {}
        # Sprint 2 : filtres par juridical_system selon les entités détectées
        return filters

    # ── Nœud 5 : GÉNÉRER ─────────────────────────────────────────────────────

    async def _generer(self, state: OrchestratorState) -> OrchestratorState:
        """
        GÉNÉRER : construire la réponse LLM avec citations obligatoires (Algo 4).
        """
        t0 = time.perf_counter()

        # Cas de refus directs (sans LLM)
        if "REFUSE_INJECTION" in state.action_plan:
            state.response = self._build_refusal_response(
                state, "security_violation", state.language
            )
            state.step_latencies["generer"] = (time.perf_counter() - t0) * 1000
            return state

        if "REFUSE_OUT_OF_SCOPE" in state.action_plan:
            state.response = self._build_refusal_response(
                state, "out_of_scope", state.language
            )
            state.step_latencies["generer"] = (time.perf_counter() - t0) * 1000
            return state

        if "ANSWER_CHITCHAT" in state.action_plan:
            state.response = await self._build_chitchat_response(state)
            state.step_latencies["generer"] = (time.perf_counter() - t0) * 1000
            return state

        # Génération LLM
        try:
            generation_service = self._get_generation_service()
            # Fusionner contexte de session + contexte KG (s'il existe)
            combined_context = self._combine_context(state)
            state.response = await generation_service.generate_answer(
                query=state.request.query,
                retrieved_chunks=state.retrieved_chunks,
                language=state.language,
                profile=state.profile,
                session_context=combined_context,
                confidence_score=state.retrieval_score,
                survey_outline=state.survey_outline,
                index_pages=state.survey_index_pages,
                web_context=state.web_context_str,
            )
            if state.web_results and state.response is not None:
                # Les sources web accompagnent la réponse sans se mêler aux
                # citations du corpus : deux statuts probatoires distincts.
                from app.models.schemas import WebSource
                state.response = state.response.model_copy(update={
                    "web_sources": [WebSource(**r) for r in state.web_results],
                })
        except Exception as exc:
            logger.error(
                "generation_failed",
                trace_id=state.trace_id,
                error=str(exc),
                exc_info=True,
            )
            state.response = self._build_refusal_response(
                state, "generation_error", state.language
            )
            state.warnings.append(f"Erreur de génération : {exc}")

        state.step_latencies["generer"] = (time.perf_counter() - t0) * 1000
        return state

    # ── Nœud 6 : VÉRIFIER ────────────────────────────────────────────────────

    async def _verifier(self, state: OrchestratorState) -> OrchestratorState:
        """
        VÉRIFIER : Algo 5 — score_conf < τ_conf → escalade.

        Conditions :
          - score < τ_esc (0.3) → ESCALATION_RECOMMENDED + avertissement fort
          - score < τ_conf (0.6) → LOW_CONFIDENCE + avertissement léger
          - absence de citations → UNSUPPORTED_CLAIMS
        """
        t0 = time.perf_counter()

        if state.response is None:
            state.step_latencies["verifier"] = (time.perf_counter() - t0) * 1000
            return state

        conf = state.retrieval_score
        state.confidence_score = conf

        extra_flags: list[SafetyFlag] = []
        extra_warnings: list[str] = []

        if conf < self.settings.escalation_threshold:
            state.escalation_needed = True
            extra_flags.append(SafetyFlag.ESCALATION_RECOMMENDED)
            if state.language == Language.EN:
                extra_warnings.append(
                    "IMPORTANT: The confidence level is very low. "
                    "Please consult a qualified professional or the relevant authority."
                )
            else:
                extra_warnings.append(
                    "IMPORTANT : Le niveau de confiance est très faible. "
                    "Veuillez consulter un professionnel qualifié ou l'autorité compétente."
                )
        elif conf < self.settings.confidence_threshold:
            extra_flags.append(SafetyFlag.LOW_CONFIDENCE)
            if state.language == Language.EN:
                extra_warnings.append(
                    "Note: This answer is based on limited evidence. Verify with official sources."
                )
            else:
                extra_warnings.append(
                    "Note : Cette réponse est fondée sur des preuves limitées. "
                    "Vérifiez auprès des sources officielles."
                )

        # Fusionner flags et warnings dans la réponse
        if extra_flags or extra_warnings or state.safety_flags:
            combined_flags = list(set(
                (state.response.safety_flags or []) + extra_flags + state.safety_flags
            ))
            # Dédoublonnage en préservant l'ordre : les avertissements de l'état
            # ont déjà pu être recopiés dans la réponse par le chemin streaming,
            # et l'utilisateur voyait le même message deux fois.
            combined_warnings = list(dict.fromkeys(
                (state.response.warnings or []) + extra_warnings + state.warnings
            ))
            state.response = state.response.model_copy(update={
                "safety_flags": combined_flags,
                "warnings": combined_warnings,
                "uncertainty_score": conf,
            })

        state.step_latencies["verifier"] = (time.perf_counter() - t0) * 1000
        logger.debug(
            "verifier_done",
            trace_id=state.trace_id,
            confidence=round(conf, 4),
            escalation=state.escalation_needed,
            flags=[f.value for f in extra_flags],
        )
        return state

    # ── Nœud 7 : ADAPTER ─────────────────────────────────────────────────────

    async def _adapter(self, state: OrchestratorState) -> OrchestratorState:
        """
        ADAPTER : écrire en mémoire épisodique, journaliser le traçage complet.

        Algo 6 (Sprint 1 simplifié) :
          - Log structuré complet (trace_id, latences, score, citations)
          - Mémoire épisodique (Sprint 3 : Neo4j / PostgreSQL sessions)
        """
        t0 = time.perf_counter()
        total_ms = (time.perf_counter() - state.start_time) * 1000

        logger.info(
            "query_trace",
            trace_id=state.trace_id,
            session_id=str(state.request.session_id),
            query_len=len(state.request.query),
            language=state.language.value,
            profile=state.profile.value,
            intent=state.intent.value,
            action_plan=state.action_plan,
            chunks_retrieved=len(state.retrieved_chunks),
            confidence=round(state.confidence_score, 4),
            escalation=state.escalation_needed,
            citations=len(state.response.citations) if state.response else 0,
            safety_flags=[f.value for f in (state.response.safety_flags or [])],
            latencies_ms=state.step_latencies,
            total_ms=round(total_ms, 2),
        )

        # Persistance PostgreSQL (Sprint 2)
        try:
            await self._persist_session(state, round(total_ms, 2))
        except Exception as exc:
            logger.warning("session_persist_failed", error=str(exc))

        state.step_latencies["adapter"] = (time.perf_counter() - t0) * 1000
        return state

    async def _persist_session(self, state: OrchestratorState, latency_ms: float) -> None:
        """Écrit un QueryLog en PostgreSQL (Algo 6, Sprint 2)."""
        from app.storage.postgres_client import get_db_session
        from app.models.db_models import QueryLog, Session as DBSession

        session_id_str = str(state.request.session_id)

        async with get_db_session() as db:
            # Upsert session (créer si inconnue)
            from sqlalchemy import select
            existing = await db.execute(
                select(DBSession).where(DBSession.id == session_id_str)
            )
            if existing.scalar_one_or_none() is None:
                db.add(DBSession(
                    id=session_id_str,
                    language=state.language.value,
                    profile_type=state.profile.value,
                ))

            # Créer le QueryLog
            db.add(QueryLog(
                session_id=session_id_str,
                query=state.request.query,
                language=state.language.value,
                profile=state.profile.value,
                intent=state.intent.value,
                plan_json={"action_plan": state.action_plan},
                retrieved_chunks_json=[
                    {"chunk_id": c.chunk_id, "source": c.source, "score": c.final_score}
                    for c in state.retrieved_chunks
                ],
                # La réponse intégrale est conservée : un aperçu de 200 caractères
                # ne permet pas de rouvrir une conversation dans l'interface.
                response_json={
                    "answer": (state.response.answer if state.response else ""),
                    "answer_preview": (state.response.answer[:200] if state.response else ""),
                    "citations_count": len(state.response.citations) if state.response else 0,
                },
                citations_json=[
                    c.model_dump() for c in (state.response.citations or [])
                ] if state.response else [],
                score_conf=round(state.confidence_score, 4),
                latency_ms=latency_ms,
                model_used=self.settings.llm_model,
                safety_flags=[f.value for f in (state.response.safety_flags or [])]
                              if state.response else [],
            ))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _detect_language(self, query: str) -> Language:
        """Détection de langue légère (avant chargement de langdetect)."""
        try:
            from app.utils.language_detection import detect_language
            return detect_language(query)
        except Exception:
            # Heuristique basique : caractères accentués → FR
            fr_chars = sum(1 for c in query if c in "àâäéèêëîïôùûüçœæÀÂÄÉÈÊËÎÏÔÙÛÜÇŒÆ")
            return Language.FR if fr_chars > 0 else Language.EN

    def _extract_entities(self, query: str) -> list[str]:
        """
        Extraction légère d'entités nommées pertinentes au droit camerounais.
        Sprint 2 : remplacé par NER (CamBERT ou spaCy).
        """
        entities: list[str] = []

        # Références légales : "article X", "loi de XXXX", "décret n°XXXX"
        article_match = re.findall(r"\bartice?l[e]?\s+\d+\b", query, re.IGNORECASE)
        entities.extend(article_match)

        law_match = re.findall(
            r"\b(?:loi|décret|arrêté|ordonnance|code)\s+(?:n[°o°]\s*)?[\w/-]+",
            query, re.IGNORECASE,
        )
        entities.extend(m.strip() for m in law_match)

        # Régions camerounaises (bijuridisme)
        regions = re.findall(
            r"\b(?:Nord-Ouest|Sud-Ouest|Northwest|Southwest|NW|SW|"
            r"Centre|Littoral|Ouest|Nord|Adamaoua|Est|Sud)\b",
            query, re.IGNORECASE,
        )
        entities.extend(regions)

        return list(set(entities))[:10]  # Max 10 entités

    def _classify_intent(self, query: str, language: Language) -> IntentType:
        """Classification par mots-clés (Sprint 2 : modèle de classification)."""
        if _OUT_OF_SCOPE_PATTERNS.search(query):
            return IntentType.OUT_OF_SCOPE

        # Bavardage : testé sur la requête entière, avant tout scoring par
        # mots-clés — sinon « bonjour » déclencherait une recherche documentaire.
        if _CHITCHAT_PATTERN.match(query.strip()):
            return IntentType.CHITCHAT

        # Agrégatif : prioritaire sur les autres intentions, car « résume le Code
        # pénal » contient « code » et serait classé NORMATIVE à tort.
        if _AGGREGATIVE_PATTERN.search(query):
            return IntentType.AGGREGATIVE

        # Normaliser les accents pour la comparaison (e.g. "etapes" ↔ "étapes")
        import unicodedata
        def _deaccent(s: str) -> str:
            return "".join(
                c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn"
            )

        query_norm = _deaccent(query.lower())
        keywords = (
            _INTENT_KEYWORDS_EN if language == Language.EN else _INTENT_KEYWORDS_FR
        )

        scores: dict[IntentType, int] = {intent: 0 for intent in keywords}
        for intent, kws in keywords.items():
            for kw in kws:
                if _deaccent(kw) in query_norm:
                    scores[intent] += 1

        best_intent = max(scores, key=lambda i: scores[i])
        return best_intent if scores[best_intent] > 0 else IntentType.FACTUAL

    def _build_refusal_response(
        self,
        state: OrchestratorState,
        reason: str,
        language: Language,
    ) -> QueryResponse:
        """Construit une réponse de refus structurée."""
        messages_fr = {
            "security_violation": (
                "Cette requête a été bloquée pour des raisons de sécurité. "
                "GOV-AI 2.0 ne répond qu'aux questions relatives "
                "à l'administration publique camerounaise."
            ),
            "out_of_scope": (
                "Cette question semble hors du périmètre de GOV-AI 2.0. "
                "Je suis spécialisé dans l'administration publique camerounaise. "
                "Reformulez votre question ou consultez les services compétents."
            ),
            "generation_error": (
                "Une erreur technique est survenue lors de la génération de la réponse. "
                "Veuillez réessayer ou contacter l'assistance technique."
            ),
        }
        messages_en = {
            "security_violation": (
                "This request has been blocked for security reasons. "
                "GOV-AI 2.0 only answers questions about Cameroonian public administration."
            ),
            "out_of_scope": (
                "This question appears to be outside the scope of GOV-AI 2.0. "
                "I specialize in Cameroonian public administration. "
                "Please rephrase your question or contact the relevant authority."
            ),
            "generation_error": (
                "A technical error occurred during response generation. "
                "Please try again or contact technical support."
            ),
        }
        msgs = messages_en if language == Language.EN else messages_fr
        answer = msgs.get(reason, msgs.get("generation_error", "Erreur inconnue."))

        flags: list[SafetyFlag] = []
        if reason == "security_violation":
            flags.append(SafetyFlag.PROMPT_INJECTION_ATTEMPT)
        elif reason == "out_of_scope":
            flags.append(SafetyFlag.OUT_OF_CORPUS)

        flags.extend(state.safety_flags)

        from app.models.schemas import QueryResponse as QR
        return QR(
            answer=answer,
            citations=[],
            retrieved_chunks=[],
            uncertainty_score=0.0,
            safety_flags=list(set(flags)),
            warnings=state.warnings,
            language_detected=language,
            model_used=self.settings.llm_model,
        )

    async def _build_chitchat_response(self, state: OrchestratorState):
        """
        Répond à une salutation sans passer par le LLM ni par le retrieval.

        Deux raisons de ne pas générer ici. D'abord une recherche documentaire sur
        « bonjour » ne rapporte que du hors-sujet, que le modèle cite ensuite comme
        s'il était pertinent. Ensuite la réponse utile est déterministe : dire ce
        que le système sait faire et, surtout, ce que le corpus contient
        RÉELLEMENT — un utilisateur qui ignore que le corpus se limite à deux codes
        posera des questions auxquelles il est impossible de répondre.
        """
        from app.models.schemas import QueryResponse as QR

        try:
            from app.services.retrieval.corpus_outline import list_documents

            documents = await list_documents()
        except Exception as exc:  # le bavardage ne doit jamais faire échouer la requête
            logger.warning("chitchat_corpus_listing_failed", error=str(exc))
            documents = []

        is_en = state.language == Language.EN
        if documents:
            listing = "\n".join(
                f"- {d['source']} ({d['chunks']} passages indexés)"
                if not is_en else
                f"- {d['source']} ({d['chunks']} indexed passages)"
                for d in documents
            )
            if is_en:
                answer = (
                    "GOV-AI 2.0 answers questions on Cameroonian law and public "
                    "administration, citing the source text for every statement.\n\n"
                    f"The corpus currently holds:\n{listing}\n\n"
                    "Questions outside these documents cannot be answered from the corpus. "
                    "Ask a specific question, or request an overview of one of these texts."
                )
            else:
                answer = (
                    "GOV-AI 2.0 répond aux questions de droit et d'administration publique "
                    "du Cameroun, en citant le texte source de chaque affirmation.\n\n"
                    f"Le corpus contient actuellement :\n{listing}\n\n"
                    "Toute question sortant de ces documents ne pourra pas être traitée à "
                    "partir du corpus. Posez une question précise, ou demandez une vue "
                    "d'ensemble de l'un de ces textes."
                )
        else:
            answer = (
                "GOV-AI 2.0 answers questions on Cameroonian law and public administration. "
                "The corpus is currently empty — ingest documents before querying."
                if is_en else
                "GOV-AI 2.0 répond aux questions de droit et d'administration publique du "
                "Cameroun. Le corpus est actuellement vide — ingérez des documents avant "
                "d'interroger le système."
            )

        return QR(
            answer=answer,
            citations=[],
            retrieved_chunks=[],
            uncertainty_score=1.0,
            safety_flags=list(state.safety_flags),
            warnings=state.warnings,
            language_detected=state.language,
            model_used="deterministe",
        )

    def _build_error_response(
        self, state: OrchestratorState, exc: Exception
    ) -> QueryResponse:
        """Réponse d'erreur interne (non exposée à l'utilisateur)."""
        from app.models.schemas import QueryResponse as QR
        lang = state.language if state.language != Language.UNKNOWN else Language.FR
        if lang == Language.EN:
            answer = (
                "An unexpected error occurred. Please try again. "
                "If the problem persists, contact technical support."
            )
        else:
            answer = (
                "Une erreur inattendue est survenue. Veuillez réessayer. "
                "Si le problème persiste, contactez le support technique."
            )
        return QR(
            answer=answer,
            citations=[],
            retrieved_chunks=[],
            uncertainty_score=0.0,
            safety_flags=[SafetyFlag.ESCALATION_RECOMMENDED],
            warnings=[f"Erreur interne : {type(exc).__name__}"],
            language_detected=lang,
            model_used=self.settings.llm_model,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_orchestrator_instance: Optional[CognitiveOrchestrator] = None


def get_orchestrator() -> CognitiveOrchestrator:
    """Retourne l'instance singleton de l'orchestrateur."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = CognitiveOrchestrator()
    return _orchestrator_instance
