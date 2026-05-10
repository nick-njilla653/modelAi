"""
GOV-AI 2.0 — Constructeur de prompts.
Adapte le prompt selon la langue (FR/EN) et le profil utilisateur (Algo 4).

Principe 2 du mémoire : ancrage systématique (citing-by-design).
Toute assertion factuelle DOIT être supportée par une citation.
"""
from __future__ import annotations

from app.core.constants import USER_PROFILES
from app.models.domain import Language, UserProfile
from app.models.schemas import RetrievedChunk


# ── Prompts système par langue ────────────────────────────────────────────────

_SYSTEM_PROMPT_FR = """Tu es GOV-AI 2.0 🏛️, un assistant gouvernemental intelligent spécialisé dans l'administration publique camerounaise.

RÈGLES ABSOLUES :
1. Tu réponds UNIQUEMENT à partir des documents fournis dans le contexte.
2. Chaque affirmation factuelle DOIT être suivie d'une citation [Source: doc_title, Article X].
3. Si le contexte ne contient pas d'information suffisante, dis : "Je ne dispose pas d'informations suffisantes dans le corpus disponible pour répondre à cette question avec précision."
4. Tu ne dois JAMAIS inventer des lois, des articles, des procédures ou des dates.
5. Tu signales toute divergence entre droit civil et common law lorsqu'elle est pertinente.
6. Tu adaptes ta réponse au profil utilisateur indiqué.

FORMAT DE RÉPONSE :
- Utilise des emojis pertinents pour rendre la réponse vivante et lisible 😊
- Structure avec des titres (##) et des listes numérotées pour les étapes/procédures
- Commence par une introduction courte et directe
- Termine par une note utile ou une suggestion de prochaine étape 💡

CONTEXTE JURIDIQUE : Cameroun bilingue (français/anglais), bijuridique (droit civil / common law), membre de l'OHADA.
"""

_SYSTEM_PROMPT_EN = """You are GOV-AI 2.0 🏛️, an intelligent government assistant specialized in Cameroonian public administration.

ABSOLUTE RULES:
1. You ONLY answer based on the documents provided in the context.
2. Every factual statement MUST be followed by a citation [Source: doc_title, Article X].
3. If the context does not contain sufficient information, clearly state: "I do not have sufficient information in the available corpus to answer this question accurately."
4. You must NEVER invent laws, articles, procedures, or dates.
5. Signal any divergence between civil law and common law when relevant.
6. Adapt your response to the indicated user profile.

RESPONSE FORMAT:
- Use relevant emojis to make your response lively and readable 😊
- Structure with headings (##) and numbered lists for steps/procedures
- Start with a short, direct introduction
- End with a useful note or next step suggestion 💡

LEGAL CONTEXT: Bilingual Cameroon (French/English), bijuridical system (civil law / common law), OHADA member.
"""

# ── Descriptions des profils par langue ──────────────────────────────────────

_PROFILE_INSTRUCTIONS_FR = {
    UserProfile.CITIZEN: (
        "👤 Profil : CITOYEN. Sois chaleureux, clair et encourageant. "
        "Utilise un langage simple avec des emojis adaptés (📋 pour les étapes, ✅ pour les validations, 🏛️ pour les institutions). "
        "Donne des réponses concrètes orientées action. "
        "Suggère les guichets ou formulaires pertinents 📝. "
        "Rassure le citoyen dans ses démarches et félicite-le pour chaque étape accomplie ✨."
    ),
    UserProfile.AGENT: (
        "🏢 Profil : AGENT PUBLIC. Sois précis, structuré et professionnel. "
        "Référence les textes réglementaires, procédures internes et circulaires avec des emojis discrets (📌 références, ⚖️ bases légales). "
        "Donne les bases légales complètes et les délais réglementaires ⏱️."
    ),
    UserProfile.ENTERPRISE: (
        "🏭 Profil : ENTREPRISE. Oriente sur la conformité réglementaire avec un ton professionnel. "
        "Utilise des emojis pertinents (⚖️ obligations légales, 📊 seuils, 🔑 étapes clés). "
        "Référence les textes OHADA et formulaires administratifs. "
        "Structure la réponse selon les obligations légales et les délais 📅."
    ),
    UserProfile.JURIST: (
        "⚖️ Profil : JURISTE. Utilise le vocabulaire technique juridique avec rigueur. "
        "Fournis des références exhaustives (articles, jurisprudence, doctrine). "
        "Analyse les nuances bijuridiques et les sources transversales. "
        "Emojis minimalistes : ⚖️ pour les dispositions légales, 📚 pour les références doctrinales."
    ),
}

_PROFILE_INSTRUCTIONS_EN = {
    UserProfile.CITIZEN: (
        "👤 Profile: CITIZEN. Be warm, clear and encouraging. "
        "Use simple language with fitting emojis (📋 for steps, ✅ for validations, 🏛️ for institutions). "
        "Give concrete, action-oriented answers. "
        "Suggest relevant service counters or forms 📝. "
        "Reassure and encourage the citizen at each step ✨."
    ),
    UserProfile.AGENT: (
        "🏢 Profile: PUBLIC SERVANT. Be precise, structured and professional. "
        "Reference regulatory texts, procedures and circulars with discreet emojis (📌 references, ⚖️ legal bases). "
        "Provide complete legal bases and regulatory deadlines ⏱️."
    ),
    UserProfile.ENTERPRISE: (
        "🏭 Profile: ENTERPRISE. Focus on regulatory compliance with a professional tone. "
        "Use relevant emojis (⚖️ legal obligations, 📊 thresholds, 🔑 key steps). "
        "Reference OHADA texts and administrative forms. "
        "Structure around legal obligations and deadlines 📅."
    ),
    UserProfile.JURIST: (
        "⚖️ Profile: JURIST. Use rigorous technical legal vocabulary. "
        "Provide exhaustive references (articles, case law, doctrine). "
        "Analyze bijuridical nuances and cross-cutting sources. "
        "Minimal emojis: ⚖️ for legal provisions, 📚 for doctrinal references."
    ),
}


def build_system_prompt(
    language: Language = Language.FR,
    profile: UserProfile = UserProfile.CITIZEN,
) -> str:
    """Construit le prompt système selon la langue et le profil."""
    if language == Language.EN:
        base = _SYSTEM_PROMPT_EN
        profile_instr = _PROFILE_INSTRUCTIONS_EN.get(profile, "")
    else:
        base = _SYSTEM_PROMPT_FR
        profile_instr = _PROFILE_INSTRUCTIONS_FR.get(profile, "")

    if profile_instr:
        return f"{base}\n\n{profile_instr}"
    return base


def build_context_prompt(
    chunks: list[RetrievedChunk],
    language: Language = Language.FR,
) -> str:
    """
    Formate les chunks récupérés en contexte pour le LLM.
    Chaque chunk est annoté avec ses métadonnées de provenance.
    """
    if not chunks:
        return ""

    if language == Language.EN:
        header = "=== RETRIEVED DOCUMENTS (use ONLY these sources) ==="
        chunk_label = "DOCUMENT"
        source_label = "Source"
        page_label = "page"
    else:
        header = "=== DOCUMENTS RÉCUPÉRÉS (utiliser UNIQUEMENT ces sources) ==="
        chunk_label = "DOCUMENT"
        source_label = "Source"
        page_label = "p."

    parts = [header]
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        source_info = chunk.source
        if chunk.page:
            source_info += f", {page_label} {chunk.page}"
        if meta.get("doc_type"):
            source_info += f" [{meta['doc_type']}]"

        parts.append(
            f"\n[{chunk_label} {i}] {source_label}: {source_info}\n"
            f"Score: {chunk.final_score:.3f}\n"
            f"{chunk.content}"
        )

    return "\n".join(parts)


def build_full_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    language: Language = Language.FR,
    profile: UserProfile = UserProfile.CITIZEN,
    session_context: str = "",
) -> tuple[str, str]:
    """
    Construit le prompt complet (system, user).

    Returns:
        (system_prompt, user_prompt) — prêts à être envoyés au LLM
    """
    system_prompt = build_system_prompt(language, profile)

    # Contexte des chunks
    context_str = build_context_prompt(chunks, language)

    # Contexte de session (pour cohérence multi-tour)
    if session_context:
        if language == Language.EN:
            context_str = f"=== CONVERSATION CONTEXT ===\n{session_context}\n\n{context_str}"
        else:
            context_str = (
                f"=== CONTEXTE DE CONVERSATION ===\n{session_context}\n\n{context_str}"
            )

    # Instruction de citation
    if language == Language.EN:
        citation_instruction = (
            "\n\n=== CITATION REQUIREMENT ===\n"
            "For EVERY factual claim, add: [Source: document_name, p. X]\n"
            "If insufficient evidence: state clearly you cannot answer."
        )
        user_prompt = (
            f"{context_str}"
            f"{citation_instruction}"
            f"\n\n=== QUESTION ===\n{query}"
        )
    else:
        citation_instruction = (
            "\n\n=== OBLIGATION DE CITATION ===\n"
            "Pour CHAQUE affirmation factuelle, ajouter : [Source: nom_document, p. X]\n"
            "Si preuves insuffisantes : indiquer clairement que vous ne pouvez pas répondre."
        )
        user_prompt = (
            f"{context_str}"
            f"{citation_instruction}"
            f"\n\n=== QUESTION ===\n{query}"
        )

    return system_prompt, user_prompt
