"""
Web Search Tool spécialisé pour les sites officiels camerounais
Remplace/améliore le WebAgentService existant avec focus sur .cm et sites gouvernementaux
"""

import re
import logging
import time
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import quote_plus, urljoin
import json

logger = logging.getLogger(__name__)

@dataclass
class WebSearchResult:
    """Résultat de recherche web"""
    title: str
    url: str
    snippet: str
    source_type: str  # "official", "news", "academic", "other"
    relevance_score: float = 0.0

class CameroonWebSearchService:
    """
    Service de recherche web spécialisé pour les sites camerounais officiels.
    Priorité donnée aux sites gouvernementaux et institutions officielles.
    """
    
    def __init__(self, llm_service=None, search_service=None, embedding_service=None):
        self.llm_service = llm_service
        self.search_service = search_service
        self.embedding_service = embedding_service
        
        self.priority_domains = {
            # Sites gouvernementaux principaux
            "gov.cm": "Site gouvernemental principal",
            "spm.gov.cm": "Services du Premier Ministre",
            "minfopra.gov.cm": "Ministère de la Fonction Publique",
            "minfi.gov.cm": "Ministère des Finances",
            "minjustice.gov.cm": "Ministère de la Justice",
            "mintransports.gov.cm": "Ministère des Transports",
            "mincommerce.gov.cm": "Ministère du Commerce",
            "minsante.gov.cm": "Ministère de la Santé",
            "minedub.gov.cm": "Ministère de l'Éducation de Base",
            "minesup.gov.cm": "Ministère de l'Enseignement Supérieur",

            # Institutions officielles
            "prc.cm": "Présidence de la République",
            "spm.gov": "service du premier",
            "senat.cm": "Sénat du Cameroun",
            "assemblee-nationale.cm": "Assemblée Nationale",
            "coursupreme.cm": "Cour Suprême",
            "conseilconstitutionnel.cm": "Conseil Constitutionnel",

            # Organismes spécialisés
            "dgi.cm": "Direction Générale des Impôts",
            "douanes.cm": "Douanes camerounaises",
            "cnps.cm": "Caisse Nationale de Prévoyance Sociale",
            "beac.int": "Banque des États de l'Afrique Centrale",
            "ohada.org": "Organisation pour l'Harmonisation du Droit des Affaires en Afrique",

            # Médias officiels
            "crtv.cm": "Cameroon Radio Television",
            "cameroon-tribune.cm": "Cameroon Tribune (journal officiel)",

            # Universités publiques
            "uy1.uninet.cm": "Université de Yaoundé I",
            "univ-dschang.org": "Université de Dschang",
            "univ-ngaoundere.cm": "Université de Ngaoundéré"
        }

        
        # Termes juridiques/administratifs camerounais
        self.cameroon_legal_terms = [
            "code général des impôts", "cgi cameroun", "ohada",
            "décret présidentiel", "arrêté ministériel", 
            "constitution cameroun", "assemblée nationale",
            "conseil constitutionnel", "cour suprême",
            "ministère", "préfecture", "sous-préfecture",
            "commune", "région cameroun", "irpp", "tva cameroun"
        ]
        
        logger.info("🇨🇲 CameroonWebSearchService initialisé")

    def needs_web_search(self, query: str) -> bool:
        """Détermine si une recherche web est nécessaire"""
        web_indicators = [
            "récent", "nouveau", "2024", "2025", "actualité",
            "prix", "taux", "statistique", "dernière",
            "mise à jour", "modification", "amendement",
            "recent", "new", "latest", "current", "update"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in web_indicators)

    async def search_official_sites(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        """
        Recherche spécifiquement sur les sites officiels camerounais
        """
        results = []
        
        try:
            # Construire les requêtes pour sites officiels
            search_queries = self._build_official_search_queries(query)
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                tasks = []
                
                for search_query in search_queries[:3]:  # Limiter à 3 requêtes
                    task = self._search_with_duckduckgo(session, search_query)
                    tasks.append(task)
                
                # Exécuter les recherches en parallèle
                search_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Traiter les résultats
                for result_set in search_results:
                    if isinstance(result_set, list):
                        results.extend(result_set)
            
            # Filtrer et classer par pertinence
            filtered_results = self._filter_and_rank_results(results, query)
            
            return filtered_results[:max_results]
            
        except Exception as e:
            logger.error(f"Erreur recherche sites officiels: {e}")
            return []

    def _build_official_search_queries(self, query: str) -> List[str]:
        """Construit des requêtes optimisées pour les sites camerounais"""
        
        queries = []
        
        # 1. Requête avec site:.cm
        queries.append(f"{query} site:.cm")
        
        # 2. Requête avec sites gouvernementaux spécifiques
        gov_sites = " OR ".join([f"site:{domain}" for domain in list(self.priority_domains.keys())[:5]])
        queries.append(f"{query} ({gov_sites})")
        
        # 3. Requête avec termes camerounais
        if any(term in query.lower() for term in ["loi", "décret", "arrêté", "article"]):
            queries.append(f'"{query}" cameroun gouvernement')
        
        # 4. Requête spécialisée selon le domaine
        query_lower = query.lower()
        if any(term in query_lower for term in ["impôt", "taxe", "fiscal"]):
            queries.append(f"{query} site:minfi.gov.cm OR site:dgi.cm")
        elif any(term in query_lower for term in ["justice", "tribunal", "procès"]):
            queries.append(f"{query} site:minjustice.gov.cm OR site:cour-supreme.cm")
        elif any(term in query_lower for term in ["travail", "emploi", "fonction publique"]):
            queries.append(f"{query} site:minfopra.gov.cm")
        
        return queries

    async def _search_with_duckduckgo(self, session: aiohttp.ClientSession, query: str) -> List[WebSearchResult]:
        """Recherche avec DuckDuckGo (gratuit et sans clé API)"""
        
        try:
            # URL DuckDuckGo Instant Answer API
            encoded_query = quote_plus(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_redirect=1&no_html=1&skip_disambig=1"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_duckduckgo_results(data, query)
                else:
                    logger.warning(f"DuckDuckGo status {response.status} pour: {query}")
                    return []
                    
        except Exception as e:
            logger.error(f"Erreur DuckDuckGo search: {e}")
            return []

    def _parse_duckduckgo_results(self, data: Dict, original_query: str) -> List[WebSearchResult]:
        """Parse les résultats DuckDuckGo"""
        
        results = []
        
        try:
            # Résultats principaux
            if data.get("AbstractText") and data.get("AbstractURL"):
                results.append(WebSearchResult(
                    title=data.get("Heading", "Résultat principal"),
                    url=data.get("AbstractURL", ""),
                    snippet=data.get("AbstractText", ""),
                    source_type=self._classify_source(data.get("AbstractURL", "")),
                    relevance_score=0.9
                ))
            
            # Résultats des topics connexes
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("FirstURL"):
                    results.append(WebSearchResult(
                        title=topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else topic.get("Text", ""),
                        url=topic.get("FirstURL", ""),
                        snippet=topic.get("Text", ""),
                        source_type=self._classify_source(topic.get("FirstURL", "")),
                        relevance_score=0.7
                    ))
            
            # Answer si disponible
            if data.get("Answer"):
                results.append(WebSearchResult(
                    title=f"Réponse directe: {original_query}",
                    url=data.get("AnswerURL", ""),
                    snippet=data.get("Answer", ""),
                    source_type="direct_answer",
                    relevance_score=0.95
                ))
                
        except Exception as e:
            logger.error(f"Erreur parsing DuckDuckGo: {e}")
        
        return results

    def _classify_source(self, url: str) -> str:
        """Classifie le type de source selon l'URL"""
        
        url_lower = url.lower()
        
        # Sites officiels camerounais
        for domain in self.priority_domains.keys():
            if domain in url_lower:
                return "official"
        
        # Sites généraux camerounais
        if ".cm" in url_lower or "cameroun" in url_lower or "cameroon" in url_lower:
            if any(indicator in url_lower for indicator in ["actu", "news", "info", "journal"]):
                return "news"
            else:
                return "cameroon_site"
        
        # Sites académiques/juridiques
        if any(indicator in url_lower for indicator in [".edu", ".ac.", "university", "universite"]):
            return "academic"
        
        # OHADA et institutions régionales
        if any(indicator in url_lower for indicator in ["ohada", "beac", "cemac"]):
            return "regional_institution"
        
        return "other"

    def _filter_and_rank_results(self, results: List[WebSearchResult], query: str) -> List[WebSearchResult]:
        """Filtre et classe les résultats par pertinence pour le Cameroun"""
        
        # Filtrer les résultats vides
        valid_results = [r for r in results if r.title and r.url and r.snippet]
        
        # Calculer le score de pertinence amélioré
        for result in valid_results:
            score = result.relevance_score
            
            # Bonus pour sites officiels
            if result.source_type == "official":
                score += 0.3
            elif result.source_type == "cameroon_site":
                score += 0.2
            elif result.source_type == "regional_institution":
                score += 0.15
            
            # Bonus pour contenu pertinent
            content_lower = (result.title + " " + result.snippet).lower()
            query_lower = query.lower()
            
            # Correspondance directe avec la requête
            if query_lower in content_lower:
                score += 0.2
            
            # Termes juridiques camerounais
            for term in self.cameroon_legal_terms:
                if term in content_lower:
                    score += 0.1
                    break
            
            # Mots-clés officiels
            official_keywords = ["ministère", "gouvernement", "officiel", "loi", "décret", "arrêté"]
            for keyword in official_keywords:
                if keyword in content_lower:
                    score += 0.05
            
            result.relevance_score = min(score, 1.0)  # Plafonner à 1.0
        
        # Trier par score décroissant
        valid_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Dédupliquer par URL
        seen_urls = set()
        unique_results = []
        for result in valid_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Interface principale compatible avec l'ancien WebAgentService
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Recherche web Cameroun: {query}")
            
            # Vérifier si web search est nécessaire
            if not self.needs_web_search(query):
                logger.info("❌ Recherche web non nécessaire pour cette requête")
                return {
                    "response": "Cette requête ne nécessite pas de recherche web récente.",
                    "web_search_performed": False,
                    "strategy": "no_web_needed",
                    "processing_time": time.time() - start_time
                }
            
            # Effectuer la recherche
            search_results = await self.search_official_sites(query, max_results=5)
            
            if not search_results:
                return {
                    "response": "Aucun résultat trouvé sur les sites officiels camerounais.",
                    "web_search_performed": True,
                    "web_sources": 0,
                    "strategy": "web_search_empty",
                    "processing_time": time.time() - start_time
                }
            
            # Générer une réponse synthétique avec le LLM
            response = await self._generate_web_response(query, search_results)
            
            # Formater les sources
            sources = {
                "web": [
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "source_type": result.source_type,
                        "relevance": result.relevance_score
                    }
                    for result in search_results
                ],
                "vector": []  # Pas de recherche vectorielle dans ce contexte
            }
            
            return {
                "response": response,
                "web_search_performed": True,
                "web_sources": len(search_results),
                "sources": sources,
                "strategy": "cameroon_official_sites",
                "processing_time": time.time() - start_time,
                "analysis": {
                    "query_type": "web_recent_info",
                    "priority_domains_found": self._count_priority_domains(search_results),
                    "official_sources": len([r for r in search_results if r.source_type == "official"])
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur process_query: {e}")
            return {
                "response": f"Erreur lors de la recherche web: {str(e)}",
                "web_search_performed": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    async def _generate_web_response(self, query: str, results: List[WebSearchResult]) -> str:
        """Génère une réponse synthétique basée sur les résultats web"""
        
        if not self.llm_service:
            # Réponse simple sans LLM
            response_parts = [f"🌐 **Informations récentes trouvées sur les sites camerounais officiels:**\n"]
            
            for i, result in enumerate(results[:3], 1):
                source_indicator = "🏛️" if result.source_type == "official" else "📰"
                response_parts.append(
                    f"{source_indicator} **{i}. {result.title}**\n"
                    f"   {result.snippet}\n"
                    f"   *Source: {result.url}*\n"
                )
            
            return "\n".join(response_parts)
        
        # Réponse avec LLM
        try:
            # Construire le contexte
            context_parts = []
            for result in results[:3]:
                context_parts.append(f"**{result.title}** ({result.source_type}): {result.snippet}")
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""Basé sur ces informations récentes trouvées sur les sites officiels camerounais, répondez à la question: "{query}"

SOURCES OFFICIELLES:
{context}

INSTRUCTIONS:
- Répondez en français
- Synthétisez les informations trouvées
- Mentionnez que les informations proviennent de sites officiels camerounais
- Soyez précis et professionnel
- Utilisez des emojis pour structurer la réponse

RÉPONSE:"""

            response = self.llm_service.generate_response(prompt, max_length=800)
            
            if isinstance(response, dict):
                return response.get("generated_text", response.get("response", ""))
            return str(response)
            
        except Exception as e:
            logger.error(f"Erreur génération réponse LLM: {e}")
            # Fallback sans LLM
            return await self._generate_web_response(query, results)

    def _count_priority_domains(self, results: List[WebSearchResult]) -> int:
        """Compte le nombre de domaines prioritaires trouvés"""
        priority_count = 0
        for result in results:
            for domain in self.priority_domains.keys():
                if domain in result.url.lower():
                    priority_count += 1
                    break
        return priority_count

    async def __aenter__(self):
        """Support du context manager async"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage des ressources"""
        pass

# Factory function pour intégration facile
def create_cameroon_web_search_service(llm_service=None, search_service=None, embedding_service=None):
    """Factory pour créer le service web camerounais"""
    return CameroonWebSearchService(
        llm_service=llm_service,
        search_service=search_service, 
        embedding_service=embedding_service
    )