"""
Service Web Agent AMÉLIORÉ pour Gov-AI
Version corrigée avec focus sites camerounais + Google fallback
"""

import asyncio
import aiohttp
import time
import random
import logging
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source_type: str = "web"  # "official", "news", "academic", "web"

class WebAgentService:
    """Agent web amélioré: sites camerounais prioritaires + Google fallback"""
    
    def __init__(self, llm_service, search_service, embedding_service):
        self.llm_service = llm_service
        self.search_service = search_service
        self.embedding_service = embedding_service
        self.session = None
        
        # Headers anti-détection
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'DNT': '1'
        }
        
        # Sites camerounais prioritaires
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
        
        # Cache simple (30 minutes)
        self.cache = {}
        self.cache_duration = 1800

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=15)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def needs_web_search(self, query: str) -> bool:
        """Détection améliorée pour recherche web"""
        query_lower = query.lower()
        
        # Mots-clés récents
        recent_keywords = [
            '2024', '2025', 'récent', 'nouveau', 'dernier', 'actualité',
            'prix', 'cours', 'taux', 'tarif', 'aujourd\'hui', 'maintenant', 
            'actuel', 'mise à jour', 'modification', 'amendement',
            'recent', 'new', 'latest', 'current', 'update'
        ]
        
        return any(keyword in query_lower for keyword in recent_keywords)

    async def search_cameroon_sites(self, query: str) -> List[WebSearchResult]:
        """Recherche prioritaire sur sites camerounais"""
        results = []
        
        # Construire requêtes spécialisées
        cameroon_queries = [
            f"{query} site:.cm",
            f"{query} site:gov.cm",
            f"{query} cameroun gouvernement"
        ]
        
        for search_query in cameroon_queries[:2]:  # Limiter pour éviter rate limiting
            try:
                google_results = await self.search_google(search_query)
                
                # Filtrer et classer par priorité camerounaise
                for result in google_results:
                    result.source_type = self._classify_cameroon_source(result.url)
                    if result.source_type in ["official", "cameroon_site"]:
                        results.append(result)
                
                if results:  # Si on trouve des résultats camerounais, arrêter
                    break
                    
            except Exception as e:
                logger.error(f"Erreur recherche Cameroun: {e}")
                continue
        
        return results[:5]

    def _classify_cameroon_source(self, url: str) -> str:
        """Classifie la source selon sa pertinence camerounaise"""
        url_lower = url.lower()
        
        # Sites officiels camerounais
        for domain in self.priority_domains:
            if domain in url_lower:
                return "official"
        
        # Sites camerounais généraux
        if ".cm" in url_lower or "cameroun" in url_lower or "cameroon" in url_lower:
            if any(indicator in url_lower for indicator in ["actu", "news", "info", "journal"]):
                return "news"
            else:
                return "cameroon_site"
        
        return "web"

    async def search_google(self, query: str) -> List[WebSearchResult]:
        """Recherche Google avec cache"""
        
        # Vérifier cache
        cache_key = f"google_{query}"
        if cache_key in self.cache:
            cache_time, results = self.cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                logger.info(f"📦 Cache hit pour: {query}")
                return results
        
        # Construire URL de recherche avec focus Cameroun
        search_url = f"https://www.google.com/search?q={query}&hl=fr&gl=cm&num=8"
        
        try:
            # Délai anti-bot
            await asyncio.sleep(random.uniform(1, 2))
            
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    html = await response.text()
                    results = self._parse_google_results(html)
                    
                    # Mettre en cache
                    self.cache[cache_key] = (time.time(), results)
                    
                    logger.info(f"🔍 Google: {len(results)} résultats pour '{query}'")
                    return results
                else:
                    logger.warning(f"⚠️ Google failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Erreur Google: {e}")
        
        return []

    def _parse_google_results(self, html: str) -> List[WebSearchResult]:
        """Parse amélioré des résultats Google"""
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        # Chercher différents sélecteurs de résultats Google
        result_selectors = [
            'div.g',           # Sélecteur standard
            'div[data-ved]',   # Sélecteur alternatif
            'div.rc'           # Ancien sélecteur
        ]
        
        result_divs = []
        for selector in result_selectors:
            result_divs = soup.select(selector)
            if result_divs:
                break
        
        for div in result_divs[:8]:
            try:
                # Titre - plusieurs possibilités
                title_elem = (div.find('h3') or 
                             div.find('a').find('h3') if div.find('a') else None)
                title = title_elem.get_text(strip=True) if title_elem else ""
                
                # Lien
                link_elem = div.find('a')
                url = link_elem.get('href') if link_elem else ""
                
                # Nettoyer URL Google
                if url and url.startswith('/url?q='):
                    url = url.split('&')[0].replace('/url?q=', '')
                
                # Snippet - plusieurs possibilités
                snippet_selectors = [
                    'span.aCOpRe', 'div.VwiC3b', 'span.st', 'div.s', 'span'
                ]
                snippet = ""
                for selector in snippet_selectors:
                    snippet_elem = div.select_one(selector)
                    if snippet_elem:
                        snippet = snippet_elem.get_text(strip=True)
                        break
                
                # Valider et ajouter
                if (title and url and 
                    not url.startswith('http://webcache') and
                    'google.com' not in url and
                    len(title) > 3):
                    
                    results.append(WebSearchResult(
                        title=title,
                        url=url,
                        snippet=snippet[:250] + "..." if len(snippet) > 250 else snippet,
                        source_type=self._classify_cameroon_source(url)
                    ))
                    
            except Exception as e:
                logger.debug(f"Erreur parsing résultat: {e}")
                continue
        
        return results

    async def search_vector_database(self, query: str) -> List[Dict]:
        """Recherche dans la base vectorielle"""
        try:
            results = self.search_service.search(query, top_k=5)
            logger.info(f"📚 Vector: {len(results)} résultats")
            return results
        except Exception as e:
            logger.error(f"❌ Erreur vector: {e}")
            return []

    async def synthesize_response(self, query: str, web_results: List[WebSearchResult], 
                                 vector_results: List[Dict]) -> str:
        """Synthèse intelligente avec priorité aux sources camerounaises"""
        
        # Trier les résultats web par pertinence camerounaise
        web_results.sort(key=lambda x: {
            "official": 3, 
            "cameroon_site": 2, 
            "news": 1, 
            "web": 0
        }.get(x.source_type, 0), reverse=True)
        
        # Contexte web avec priorité camerounaise
        web_context = ""
        if web_results:
            web_context = "🌐 **INFORMATIONS RÉCENTES :**\n"
            for i, result in enumerate(web_results[:3], 1):
                source_emoji = {
                    "official": "🏛️",
                    "cameroon_site": "🇨🇲", 
                    "news": "📰",
                    "web": "🌐"
                }.get(result.source_type, "🌐")
                
                web_context += f"{source_emoji} **{i}. {result.title}**\n"
                web_context += f"   {result.snippet}\n"
                web_context += f"   *Source: {result.url}*\n\n"
        
        # Contexte vectoriel (base juridique)
        vector_context = ""
        if vector_results:
            vector_context = "📚 **BASE JURIDIQUE CAMEROUNAISE :**\n"
            for i, result in enumerate(vector_results[:3], 1):
                content = result.get("content", result.get("text", ""))[:300]
                metadata = result.get("metadata", {})
                source = metadata.get("filename", metadata.get("source", "Document juridique"))
                page = metadata.get("page_number", "")
                page_info = f" (page {page})" if page else ""
                
                vector_context += f"📄 **{i}. {source}**{page_info}\n"
                vector_context += f"   {content}...\n\n"
        
        # Prompt optimisé pour Gov-AI
        prompt = f"""Vous êtes Gov-AI, assistant conversationnel du Cameroun. Répondez à cette question en synthétisant intelligemment les informations disponibles.

**Question :** {query}

{vector_context}
{web_context}

**INSTRUCTIONS :**
- Répondez en français de manière naturelle et engageante
- Privilégiez les sources officielles camerounaises (🏛️) 
- Utilisez d'abord la base juridique, complétez avec le web si pertinent
- Intégrez les informations de manière fluide (pas de "selon le web" vs "selon la base")
- Mentionnez les sources importantes naturellement dans le texte
- Adoptez un ton conversationnel avec des emojis occasionnels
- Soyez précis mais accessible

**Réponse :**"""

        try:
            response = self.llm_service.generate_response(prompt, max_length=1000)
            
            if isinstance(response, dict):
                return response.get("generated_text", response.get("response", ""))
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"❌ Erreur LLM synthesis: {e}")
            
            # Fallback simple sans LLM
            if web_results and vector_results:
                return f"Voici ce que j'ai trouvé sur '{query}':\n\n{web_context[:200]}...\n\n{vector_context[:200]}..."
            elif web_results:
                return f"Informations récentes trouvées:\n\n{web_context[:300]}..."
            elif vector_results:
                return f"D'après la base juridique camerounaise:\n\n{vector_context[:300]}..."
            else:
                return "Je n'ai pas trouvé d'informations pertinentes sur cette question."

    async def process_query(self, query: str) -> Dict:
        """Point d'entrée principal avec stratégie camerounaise"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Traitement: {query}")
            
            # 1. Vérifier si web nécessaire
            needs_web = self.needs_web_search(query)
            logger.info(f"🌐 Recherche web nécessaire: {needs_web}")
            
            # 2. Recherche vectorielle (toujours en premier)
            vector_results = await self.search_vector_database(query)
            
            # 3. Recherche web si nécessaire
            web_results = []
            strategy = "vector_only"
            
            if needs_web:
                # Essayer d'abord sites camerounais
                web_results = await self.search_cameroon_sites(query)
                
                if not web_results:
                    # Fallback Google général
                    web_results = await self.search_google(query)
                    strategy = "google_fallback"
                else:
                    strategy = "cameroon_priority"
            
            # 4. Synthèse intelligente
            response = await self.synthesize_response(query, web_results, vector_results)
            
            # 5. Statistiques
            official_sources = len([r for r in web_results if r.source_type == "official"])
            cameroon_sources = len([r for r in web_results if r.source_type in ["official", "cameroon_site"]])
            
            return {
                "query": query,
                "response": response,
                "web_search_performed": needs_web,
                "web_sources": len(web_results),
                "vector_sources": len(vector_results),
                "processing_time": time.time() - start_time,
                "strategy": strategy,
                "analysis": {
                    "official_sources": official_sources,
                    "cameroon_sources": cameroon_sources,
                    "web_needed": needs_web
                },
                "sources": {
                    "web": [
                        {
                            "title": r.title, 
                            "url": r.url, 
                            "type": r.source_type,
                            "snippet": r.snippet[:100] + "..." if len(r.snippet) > 100 else r.snippet
                        } 
                        for r in web_results
                    ],
                    "vector": [
                        {
                            "source": r.get("metadata", {}).get("filename", "Document"),
                            "page": r.get("metadata", {}).get("page_number", "")
                        } 
                        for r in vector_results[:3]
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur process_query: {e}")
            return {
                "query": query,
                "response": f"Désolé, j'ai rencontré un problème technique. Pouvez-vous reformuler votre question ?",
                "web_search_performed": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "strategy": "error"
            }