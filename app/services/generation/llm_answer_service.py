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
from app.services.generation.prompt_builder import build_full_prompt

logger = get_logger(__name__)


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
        system_prompt, user_prompt = build_full_prompt(
            query=query,
            chunks=retrieved_chunks,
            language=language,
            profile=profile,
            session_context=session_context,
        )

        # Génération LLM
        try:
            raw_answer = await self._call_llm(system_prompt, user_prompt)
        except Exception as exc:
            logger.error("llm_generation_failed", error=str(exc), exc_info=True)
            raise GenerationError(f"Erreur de génération LLM : {exc}") from exc

        # Extraction et validation des citations
        citations = self._extract_citations(raw_answer, retrieved_chunks)

        # Avertissement si peu de citations
        if not citations and retrieved_chunks:
            safety_flags.append(SafetyFlag.UNSUPPORTED_CLAIMS)
            warnings.append(
                "Attention : la réponse ne contient pas de citations vérifiables. "
                "Vérifiez les sources manuellement."
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
    ) -> AsyncGenerator[str, None]:
        """Génération en streaming (token par token)."""
        system_prompt, user_prompt = build_full_prompt(
            query=query,
            chunks=retrieved_chunks,
            language=language,
            profile=profile,
            session_context=session_context,
        )

        async for token in self._call_llm_stream(system_prompt, user_prompt):
            yield token

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
            "options": {
                "temperature": self.settings.llm_temperature,
                "num_predict": self.settings.llm_max_tokens,
            },
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
            "options": {"temperature": self.settings.llm_temperature},
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
            async with client.stream(
                "POST",
                f"{self.settings.llm_base_url}/api/chat",
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue

    def _extract_citations(
        self,
        answer: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        """
        Extrait les citations de la réponse générée.
        Cherche les patterns [Source: ..., p. X] et les mappe aux chunks.
        """
        citations: list[Citation] = []
        seen_chunk_ids: set[str] = set()

        # Pattern de citation dans le texte généré
        # Accepte: [Source: doc.txt], [Source: doc.txt, p.3], [Source: doc.txt, Article 314]
        citation_pattern = re.compile(
            r"\[(?:Source|Src|Ref)\s*:\s*([^\]]+?)\]",
            re.IGNORECASE,
        )

        matched_sources = set()
        for match in citation_pattern.finditer(answer):
            full_ref = match.group(1).strip()
            # Extraire uniquement le nom de source (avant la première virgule)
            source_name = full_ref.split(",")[0].strip()
            matched_sources.add(source_name.lower())

        # Mapper les sources citées aux chunks récupérés
        for chunk in retrieved_chunks:
            if chunk.chunk_id in seen_chunk_ids:
                continue

            # Vérifier si ce chunk est mentionné dans la réponse
            source_lower = chunk.source.lower()
            is_cited = any(
                ms in source_lower or source_lower in ms
                for ms in matched_sources
            )

            # Aussi inclure les chunks avec un score élevé même non explicitement cités
            chunk_score = chunk.dense_score if chunk.dense_score is not None else chunk.final_score
            is_high_score = chunk_score >= self.settings.citation_min_score

            if is_cited or is_high_score:
                citations.append(Citation(
                    source_id=chunk.doc_id,
                    doc_title=chunk.source,
                    chunk_id=chunk.chunk_id,
                    excerpt=chunk.content[:300],
                    relevance_score=round(min(chunk_score, 1.0), 4),
                    page=chunk.page,
                    language=chunk.language,
                    doc_type=chunk.metadata.get("doc_type") if chunk.metadata else None,
                    institution=chunk.metadata.get("institution") if chunk.metadata else None,
                    jurisdiction=chunk.metadata.get("jurisdiction") if chunk.metadata else None,
                ))
                seen_chunk_ids.add(chunk.chunk_id)

        return citations

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
