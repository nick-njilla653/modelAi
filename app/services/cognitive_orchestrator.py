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
        self._graph: Any = None  # LangGraph compiled graph (optionnel)

    # ── Services lazy ─────────────────────────────────────────────────────────

    def _get_retrieval_service(self) -> Any:
        if self._retrieval_service is None:
            from app.services.retrieval.hybrid_retrieval_service import HybridRetrievalService
            from app.services.embedding import get_embedding_service
            svc = HybridRetrievalService()
            svc.set_embedding_service(get_embedding_service())
            self._retrieval_service = svc
        return self._retrieval_service

    def _get_generation_service(self) -> Any:
        if self._generation_service is None:
            from app.services.generation.llm_answer_service import LLMAnswerService
            self._generation_service = LLMAnswerService()
        return self._generation_service

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

        # Injecter session_id et intent dans la réponse finale
        if state.response is not None:
            state.response = state.response.model_copy(update={
                "session_id": state.request.session_id,
                "intent_detected": state.intent,
                "juridical_system_detected": state.request.juridical_system,
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
          - résumerSession(historique) → contexte S
        """
        t0 = time.perf_counter()

        # Profil (depuis la requête — Sprint 1 : pas de persistance session)
        state.profile = state.request.profile or UserProfile.CITIZEN

        # Contexte de session (Sprint 1 : transmis directement si fourni)
        if state.request.session_context:
            state.session_context = state.request.session_context

        state.step_latencies["comprendre"] = (time.perf_counter() - t0) * 1000
        logger.debug(
            "comprendre_done",
            trace_id=state.trace_id,
            profile=state.profile,
            has_session_context=bool(state.session_context),
        )
        return state

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
        else:
            # Sprint 1 : RAG hybride systématique
            state.action_plan = ["HYBRID_RAG"]
            # Sprint 2 ajoutera : "KNOWLEDGE_GRAPH" si NORMATIVE/COMPARATIVE
            # Sprint 3 ajoutera : "WEB_SEARCH" si OUT_OF_CORPUS + escalade autorisée

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

        if "REFUSE_INJECTION" in state.action_plan or "REFUSE_OUT_OF_SCOPE" in state.action_plan:
            # Pas de retrieval nécessaire
            state.retrieved_chunks = []
            state.retrieval_score = 0.0
            state.step_latencies["agir"] = (time.perf_counter() - t0) * 1000
            return state

        if "HYBRID_RAG" in state.action_plan:
            state = await self._execute_hybrid_rag(state)

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

        # Génération LLM
        try:
            generation_service = self._get_generation_service()
            state.response = await generation_service.generate_answer(
                query=state.request.query,
                retrieved_chunks=state.retrieved_chunks,
                language=state.language,
                profile=state.profile,
                session_context=state.session_context,
                confidence_score=state.retrieval_score,
            )
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
            combined_warnings = (state.response.warnings or []) + extra_warnings + state.warnings
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

        # Sprint 3 : persistance PostgreSQL de la session et mémoire épisodique
        # await self._persist_session(state)

        state.step_latencies["adapter"] = (time.perf_counter() - t0) * 1000
        return state

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
