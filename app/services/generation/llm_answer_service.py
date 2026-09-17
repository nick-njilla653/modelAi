"""
GOV-AI 2.0 — Service de génération LLM avec citations obligatoires.
Implémente l'Algorithme 4 du mémoire (§4.3.4) : génération contrainte par citations.

Principe : réponses UNIQUEMENT à partir des preuves récupérées.
Refus si score de confiance < τ_conf (Algorithme 5).
"""
from __future__ import annotations

import re
import time
from typing import Any, AsyncGenerator, Optional

from app.core.config import get_settings
from app.core.exceptions import GenerationError, InsufficientEvidenceError
from app.core.logging import get_logger
from app.models.domain import Language, SafetyFlag, UserProfile
from app.models.schemas import Citation, QueryResponse, RetrievedChunk
from app.services.generation.prompt_builder import build_full_prompt, build_survey_prompt

logger = get_logger(__name__)


# ── Résolution des références citées ──────────────────────────────────────────
# Accepte : [Source: doc.pdf], [Source: doc.pdf, p. 3], [Source: doc.pdf, art. 314, p. 3]
_CITATION_PATTERN = re.compile(
    r"\[(?:Source|Src|Ref)\s*:\s*([^\]]+?)\]",
    re.IGNORECASE,
)
_PAGE_IN_REF_PATTERN = re.compile(r"\bp(?:age|\.|\b)\s*(\d+)", re.IGNORECASE)
_SOURCE_NOISE_PATTERN = re.compile(r"[^a-z0-9]+")
_SOURCE_EXTENSIONS = (".pdf", ".txt", ".docx", ".doc", ".html", ".md")


def _normalize_source(name: str) -> str:
    """Normalise un nom de document pour comparaison (casse, extension, ponctuation)."""
    normalized = name.strip().lower()
    for ext in _SOURCE_EXTENSIONS:
        if normalized.endswith(ext):
            normalized = normalized[: -len(ext)]
            break
    return _SOURCE_NOISE_PATTERN.sub(" ", normalized).strip()


def _sources_match(cited: str, actual: str) -> bool:
    """
    Vrai si la référence citée désigne bien ce document.

    Tolère les variantes de nommage (« code penal » vs « Code Penal.pdf ») mais
    refuse les correspondances dégénérées sur des fragments trop courts.
    """
    if not cited or not actual:
        return False
    if cited == actual:
        return True
    if len(cited) < 4 or len(actual) < 4:
        return False
    return cited in actual or actual in cited


def _page_attested_by_index(
    cited_source: str,
    page: int,
    index_pages: Optional[dict[str, set[int]]],
) -> bool:
    """
    Vrai si la page citée figure dans l'ossature indexée du document.

    Lors d'une requête agrégative, le modèle cite spontanément le plan du
    document — « [Source: CPP, p. 3] » pour un titre situé page 3. La référence
    est exacte, mais aucun extrait ne l'accompagne : sans ce contrôle elle serait
    signalée comme une hallucination, et chaque vue d'ensemble paraîtrait fautive.
    """
    if not index_pages:
        return False
    return any(
        _sources_match(cited_source, _normalize_source(source)) and page in pages
        for source, pages in index_pages.items()
    )


class LLMAnswerService:
    """
    Service de génération LLM (Ollama / compatible OpenAI).
    Implémente la génération contrainte par citations (citing-by-design).
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    async def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        language: Language = Language.FR,
        profile: UserProfile = UserProfile.CITIZEN,
        session_context: str = "",
        confidence_score: float = 1.0,
        survey_outline: str = "",
        index_pages: Optional[dict[str, set[int]]] = None,
        web_context: str = "",
    ) -> QueryResponse:
        """
        Génère une réponse ancrée sur les chunks récupérés (Algo 4).

        Args:
            query: Question de l'utilisateur
            retrieved_chunks: Chunks récupérés après reranking
            language: Langue de génération
            profile: Profil utilisateur
            session_context: Résumé du contexte de session
            confidence_score: Score de confiance du retrieval

        Returns:
            QueryResponse avec answer, citations, flags

        Raises:
            InsufficientEvidenceError: si confidence_score < τ_conf
        """
        start = time.perf_counter()
        safety_flags: list[SafetyFlag] = []
        warnings: list[str] = []

        # Vérification des preuves (Algo 5 — appel précoce)
        if not retrieved_chunks:
            return self._build_insufficient_response(
                query, language, confidence_score, "no_chunks"
            )

        if confidence_score < self.settings.escalation_threshold:
            safety_flags.append(SafetyFlag.ESCALATION_RECOMMENDED)
            warnings.append(
                "Escalade recommandée : confiance très faible. "
                "Veuillez consulter un professionnel compétent."
            )

        # Construction des prompts (Algo 4 §4.3.4)
        if survey_outline:
            # Requête agrégative : le modèle décrit l'étendue du document d'après
            # son ossature indexée, et n'illustre qu'avec les extraits échantillonnés.
            system_prompt, user_prompt = build_survey_prompt(
                query=query,
                outline=survey_outline,
                chunks=retrieved_chunks,
                language=language,
                profile=profile,
            )
        else:
            system_prompt, user_prompt = build_full_prompt(
                query=query,
                chunks=retrieved_chunks,
                language=language,
                profile=profile,
                session_context=session_context,
                web_context=web_context,
            )

        # Génération LLM
        try:
            raw_answer = await self._call_llm(system_prompt, user_prompt)
        except Exception as exc:
            # Les exceptions de timeout httpx ont un str() vide : sans ce traitement
            # le message remonté est « Erreur de génération LLM : », inexploitable.
            import httpx

            if isinstance(exc, httpx.TimeoutException):
                detail = (
                    f"le modèle {self.settings.llm_model} n'a pas répondu dans le délai "
                    f"imparti ({self.settings.llm_timeout} s). Augmentez LLM_TIMEOUT, "
                    f"réduisez LLM_MAX_TOKENS (actuellement {self.settings.llm_max_tokens}), "
                    f"ou utilisez un modèle plus rapide."
                )
            else:
                detail = str(exc) or type(exc).__name__

            logger.error(
                "llm_generation_failed",
                error=detail,
                error_type=type(exc).__name__,
                exc_info=True,
            )
            raise GenerationError(f"Erreur de génération LLM : {detail}") from exc

        # Extraction et validation des citations
        citations, unresolved_refs = self._extract_citations(
            raw_answer, retrieved_chunks, index_pages
        )

        # Avertissement si aucune affirmation n'est appuyée
        if not citations and retrieved_chunks:
            safety_flags.append(SafetyFlag.UNSUPPORTED_CLAIMS)
            warnings.append(
                "Attention : la réponse ne contient pas de citations vérifiables. "
                "Vérifiez les sources manuellement."
            )

        # Références citées qui ne correspondent à aucun extrait fourni
        if unresolved_refs:
            if SafetyFlag.UNSUPPORTED_CLAIMS not in safety_flags:
                safety_flags.append(SafetyFlag.UNSUPPORTED_CLAIMS)
            distinct_refs = list(dict.fromkeys(unresolved_refs))[:5]
            warnings.append(
                "Références citées introuvables dans le contexte fourni : "
                + " ; ".join(distinct_refs)
            )
            logger.warning(
                "unresolved_citations",
                refs=distinct_refs,
                total=len(unresolved_refs),
            )

        latency_ms = (time.perf_counter() - start) * 1000

        return QueryResponse(
            answer=raw_answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            uncertainty_score=confidence_score,
            safety_flags=safety_flags,
            warnings=warnings,
            language_detected=language,
            latency_ms=round(latency_ms, 2),
            model_used=self.settings.llm_model,
        )

    async def generate_stream(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        language: Language = Language.FR,
        profile: UserProfile = UserProfile.CITIZEN,
        session_context: str = "",
        survey_outline: str = "",
        web_context: str = "",
    ) -> AsyncGenerator[str, None]:
        """
        Génération en streaming (token par token).

        Accepte la même ossature que `generate_answer` : sans cela une requête
        agrégative streamée retomberait sur le prompt ponctuel et perdrait le
        parcours structurel.
        """
        # Réponse de refus immédiate si pas de chunks (sans appel LLM)
        if not retrieved_chunks:
            if language == Language.EN:
                yield (
                    "I do not have sufficient information in the available corpus to answer "
                    "this question. Please ingest documents first or consult the relevant authority."
                )
            else:
                yield (
                    "Je ne dispose pas d'informations suffisantes dans le corpus pour répondre "
                    "à cette question. Ingérez d'abord des documents via l'onglet Ingestion."
                )
            return

        if survey_outline:
            system_prompt, user_prompt = build_survey_prompt(
                query=query,
                outline=survey_outline,
                chunks=retrieved_chunks,
                language=language,
                profile=profile,
            )
        else:
            system_prompt, user_prompt = build_full_prompt(
                query=query,
                chunks=retrieved_chunks,
                language=language,
                profile=profile,
                session_context=session_context,
                web_context=web_context,
            )

        async for token in self._call_llm_stream(system_prompt, user_prompt):
            yield token

    def _ollama_options(self) -> dict[str, Any]:
        """
        Options d'inférence Ollama, identiques en streaming et en non-streaming.

        num_ctx est explicite : la valeur par défaut d'Ollama tronque
        silencieusement le prompt, ce qui ferait perdre une partie des extraits
        récupérés — et donc l'ancrage des citations.
        """
        return {
            "temperature": self.settings.llm_temperature,
            "num_predict": self.settings.llm_max_tokens,
            "num_ctx": self.settings.llm_num_ctx,
            "repeat_penalty": self.settings.llm_repeat_penalty,
        }

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Appelle le LLM (Ollama ou compatible OpenAI)."""
        if self.settings.llm_provider == "ollama":
            return await self._call_ollama(system_prompt, user_prompt)
        else:
            return await self._call_openai_compatible(system_prompt, user_prompt)

    async def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Appel Ollama API (POST /api/chat)."""
        import httpx

        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": self._ollama_options(),
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
            response = await client.post(
                f"{self.settings.llm_base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]

    async def _call_openai_compatible(
        self, system_prompt: str, user_prompt: str
    ) -> str:
        """Appel API compatible OpenAI."""
        import httpx

        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
            response = await client.post(
                f"{self.settings.llm_base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def _call_llm_stream(
        self, system_prompt: str, user_prompt: str
    ) -> AsyncGenerator[str, None]:
        """Streaming depuis Ollama."""
        import json
        import httpx

        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": self._ollama_options(),
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
            async with client.stream(
                "POST",
                f"{self.settings.llm_base_url}/api/chat",
                json=payload,
            ) as response:
                if response.status_code == 404:
                    raise GenerationError(
                        f"Modèle '{self.settings.llm_model}' introuvable dans Ollama. "
                        f"Lancez : ollama pull {self.settings.llm_model}"
                    )
                if response.status_code != 200:
                    body = await response.aread()
                    raise GenerationError(
                        f"Ollama erreur {response.status_code} : {body.decode()[:200]}"
                    )
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("error"):
                            raise GenerationError(f"Ollama : {data['error']}")
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue

    def _extract_citations(
        self,
        answer: str,
        retrieved_chunks: list[RetrievedChunk],
        index_pages: Optional[dict[str, set[int]]] = None,
    ) -> tuple[list[Citation], list[str]]:
        """
        Extrait les citations de la réponse générée et les résout sur les chunks.

        Principe d'ancrage : SEUL un chunk effectivement cité dans la réponse
        devient une citation. Un chunk bien classé mais non cité n'appuie aucune
        affirmation et ne doit donc pas être présenté comme une source.

        Returns:
            (citations résolues, références citées non résolues)
            Les références non résolues signalent un texte cité qui n'existe pas
            dans le contexte fourni — donc une hallucination de référence.
        """
        citations: list[Citation] = []
        seen_chunk_ids: set[str] = set()
        unresolved: list[str] = []

        for match in _CITATION_PATTERN.finditer(answer):
            full_ref = match.group(1).strip()
            source_name = full_ref.split(",")[0].strip()
            normalized_ref = _normalize_source(source_name)
            if not normalized_ref:
                continue

            page_match = _PAGE_IN_REF_PATTERN.search(full_ref)
            cited_page = int(page_match.group(1)) if page_match else None

            # Chunks du document cité
            candidates = [
                c for c in retrieved_chunks
                if _sources_match(normalized_ref, _normalize_source(c.source))
            ]
            if not candidates:
                unresolved.append(full_ref)
                continue

            # Si la citation porte une page, ne retenir que les chunks de cette page
            if cited_page is not None:
                page_matched = [c for c in candidates if c.page == cited_page]
                if page_matched:
                    candidates = page_matched
                elif _page_attested_by_index(normalized_ref, cited_page, index_pages):
                    # Page issue de l'ossature indexée (requête agrégative) : la
                    # référence est exacte, mais aucun extrait ne l'accompagne.
                    # Ce n'est ni une citation vérifiable ni une hallucination.
                    continue
                else:
                    # Document connu, page inexistante dans le contexte fourni
                    unresolved.append(full_ref)
                    continue

            for chunk in candidates:
                if chunk.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk.chunk_id)
                citations.append(self._chunk_to_citation(chunk))

        return citations, unresolved

    def _chunk_to_citation(self, chunk: RetrievedChunk) -> Citation:
        """Convertit un chunk récupéré en citation vérifiable."""
        score = chunk.dense_score if chunk.dense_score is not None else chunk.final_score
        meta = chunk.metadata or {}

        def _optional(key: str) -> Optional[str]:
            """
            Une métadonnée absente vaut None, jamais la chaîne vide.

            Les couches de récupération remplissent les champs manquants avec
            "" pour uniformiser leurs dictionnaires. Or `doc_type` est une
            énumération : "" n'en est pas une valeur, et Pydantic rejetait la
            citation entière — la réponse échouait au lieu de s'afficher.
            """
            value = meta.get(key)
            return value if isinstance(value, str) and value.strip() else None
        return Citation(
            source_id=chunk.doc_id,
            doc_title=chunk.source,
            chunk_id=chunk.chunk_id,
            excerpt=chunk.content[:300],
            relevance_score=round(min(score, 1.0), 4),
            page=chunk.page,
            # Référence extraite à l'ingestion : c'est elle qui s'affiche dans la
            # citation, pas un numéro reconstruit par le modèle.
            article=_optional("article_ref"),
            language=chunk.language,
            doc_type=_optional("doc_type"),
            institution=_optional("institution"),
            jurisdiction=_optional("jurisdiction"),
        )

    def _build_insufficient_response(
        self,
        query: str,
        language: Language,
        confidence_score: float,
        reason: str,
    ) -> QueryResponse:
        """Construit une réponse de refus (preuves insuffisantes)."""
        if language == Language.EN:
            answer = (
                "I do not have sufficient information in the available corpus to answer "
                "this question accurately. Please consult the relevant administrative "
                "authority or a qualified professional."
            )
        else:
            answer = (
                "Je ne dispose pas d'informations suffisantes dans le corpus disponible "
                "pour répondre à cette question de manière fiable. "
                "Veuillez consulter l'autorité administrative compétente ou un professionnel qualifié."
            )

        return QueryResponse(
            answer=answer,
            citations=[],
            retrieved_chunks=[],
            uncertainty_score=confidence_score,
            safety_flags=[SafetyFlag.ESCALATION_RECOMMENDED],
            warnings=[f"Réponse refusée : {reason}"],
            language_detected=language,
            model_used=self.settings.llm_model,
        )
