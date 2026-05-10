"""
Tests unitaires — Prompt Builder (app/services/generation/prompt_builder.py).
Vérifie la construction des prompts FR/EN par profil utilisateur.
"""
import pytest

from app.models.domain import Language, UserProfile
from app.models.schemas import RetrievedChunk
from app.services.generation.prompt_builder import (
    build_system_prompt,
    build_context_prompt,
    build_full_prompt,
)


@pytest.fixture
def chunks_fr():
    return [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            content="Article 1 : La République du Cameroun est un État unitaire.",
            source="Constitution_1996.pdf",
            language="fr",
            page=1,
            chunk_index=0,
            final_score=0.90,
            metadata={"doc_type": "constitution"},
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id="d1",
            content="Article 2 : La souveraineté nationale appartient au peuple.",
            source="Constitution_1996.pdf",
            language="fr",
            page=2,
            chunk_index=1,
            final_score=0.80,
        ),
    ]


@pytest.fixture
def chunks_en():
    return [
        RetrievedChunk(
            chunk_id="c3",
            doc_id="d2",
            content="Section 1: The Republic of Cameroon is a unitary state.",
            source="Constitution_1996_EN.pdf",
            language="en",
            page=1,
            chunk_index=0,
            final_score=0.88,
            metadata={"doc_type": "constitution"},
        ),
    ]


class TestBuildSystemPrompt:

    def test_french_prompt_contains_key_rules(self):
        prompt = build_system_prompt(Language.FR, UserProfile.CITIZEN)
        assert "RÈGLES ABSOLUES" in prompt
        assert "citation" in prompt.lower() or "Citation" in prompt
        assert "GOV-AI 2.0" in prompt

    def test_english_prompt_contains_key_rules(self):
        prompt = build_system_prompt(Language.EN, UserProfile.CITIZEN)
        assert "ABSOLUTE RULES" in prompt
        assert "citation" in prompt.lower() or "Citation" in prompt
        assert "GOV-AI 2.0" in prompt

    def test_citizen_profile_fr(self):
        prompt = build_system_prompt(Language.FR, UserProfile.CITIZEN)
        assert "CITOYEN" in prompt.upper() or "citoyen" in prompt.lower()

    def test_agent_profile_fr(self):
        prompt = build_system_prompt(Language.FR, UserProfile.AGENT)
        assert "AGENT" in prompt.upper()

    def test_jurist_profile_en(self):
        prompt = build_system_prompt(Language.EN, UserProfile.JURIST)
        assert "JURIST" in prompt.upper()

    def test_enterprise_profile_en(self):
        prompt = build_system_prompt(Language.EN, UserProfile.ENTERPRISE)
        assert "ENTERPRISE" in prompt.upper() or "OHADA" in prompt


class TestBuildContextPrompt:

    def test_includes_all_chunks(self, chunks_fr):
        context = build_context_prompt(chunks_fr, Language.FR)
        assert "Article 1" in context
        assert "Article 2" in context

    def test_includes_source_info(self, chunks_fr):
        context = build_context_prompt(chunks_fr, Language.FR)
        assert "Constitution_1996.pdf" in context

    def test_includes_scores(self, chunks_fr):
        context = build_context_prompt(chunks_fr, Language.FR)
        assert "0.9" in context or "Score" in context

    def test_empty_chunks(self):
        context = build_context_prompt([], Language.FR)
        assert context == ""

    def test_en_header(self, chunks_en):
        context = build_context_prompt(chunks_en, Language.EN)
        assert "RETRIEVED DOCUMENTS" in context

    def test_fr_header(self, chunks_fr):
        context = build_context_prompt(chunks_fr, Language.FR)
        assert "DOCUMENTS RÉCUPÉRÉS" in context


class TestBuildFullPrompt:

    def test_returns_tuple(self, chunks_fr):
        result = build_full_prompt("Qu'est-ce que la Constitution?", chunks_fr, Language.FR)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_system_prompt_not_empty(self, chunks_fr):
        sys_prompt, user_prompt = build_full_prompt("test", chunks_fr, Language.FR)
        assert len(sys_prompt) > 50

    def test_user_prompt_contains_query(self, chunks_fr):
        query = "Qu'est-ce que l'article 1 ?"
        _, user_prompt = build_full_prompt(query, chunks_fr, Language.FR)
        assert query in user_prompt

    def test_user_prompt_contains_citation_instruction_fr(self, chunks_fr):
        _, user_prompt = build_full_prompt("test", chunks_fr, Language.FR)
        assert "CITATION" in user_prompt.upper() or "citation" in user_prompt.lower()

    def test_user_prompt_contains_citation_instruction_en(self, chunks_en):
        _, user_prompt = build_full_prompt("test", chunks_en, Language.EN)
        assert "CITATION" in user_prompt.upper()

    def test_session_context_included(self, chunks_fr):
        _, user_prompt = build_full_prompt(
            "test", chunks_fr, Language.FR,
            session_context="L'utilisateur a précédemment demandé sur le droit civil."
        )
        assert "CONTEXTE" in user_prompt.upper() or "contexte" in user_prompt.lower()
