# GOV-AI 2.0 — Assistant Gouvernemental Intelligent

> **Système RAG juridique bilingue et bijuridique pour l'administration publique camerounaise.**
> Souveraineté numérique complète : 100 % on-premise, aucune donnée ne quitte le périmètre CENADI.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture technique](#architecture-technique)
3. [Pipeline cognitif (7 étapes)](#pipeline-cognitif)
4. [Stack technologique](#stack-technologique)
5. [Structure du projet](#structure-du-projet)
6. [Démarrage rapide (Docker)](#démarrage-rapide)
7. [API — Endpoints principaux](#api--endpoints-principaux)
8. [Ingestion de documents](#ingestion-de-documents)
9. [Interface web](#interface-web)
10. [Évaluation (B0→B4)](#évaluation)
11. [Configuration](#configuration)
12. [Développement local](#développement-local)

---

## Vue d'ensemble

GOV-AI 2.0 est un assistant juridique conversationnel conçu pour les agents et citoyens de l'administration camerounaise. Il répond à des questions en droit administratif, civil, commercial, pénal, fiscal et OHADA, en français et en anglais, dans les deux systèmes juridiques coexistants du Cameroun (droit civil francophone / common law anglophone).

**Fonctionnalités clés :**

- Recherche hybride (dense HNSW + BM25 lexical) sur le corpus juridique national
- Re-ranking croisé par modèle de langage (BAAI/bge-reranker-v2-m3)
- Graphe de connaissances juridiques (Neo4j) pour le raisonnement inter-documents
- Mémoire de session persistante (PostgreSQL) — historique des 5 derniers échanges
- Fallback recherche web (DuckDuckGo, désactivé par défaut pour souveraineté)
- Génération streaming token-par-token via Ollama (llama3.2:latest)
- Plan d'action exposé dans la réponse API (`action_plan`)
- Détection d'intention et de système juridique (civil_law / common_law / ohada / bijuridique)

---

## Architecture technique

```
┌─────────────────────────────────────────────────────────────────┐
│                         UTILISATEUR                             │
│                    Interface Web (/ui)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP / SSE (streaming)
┌───────────────────────────▼─────────────────────────────────────┐
│                     GOV-AI 2.0 API                              │
│              FastAPI 0.115 · Python 3.11                        │
│                                                                  │
│  POST /api/v1/query          → réponse complète JSON            │
│  POST /api/v1/query/stream   → Server-Sent Events               │
│  POST /api/v1/ingest         → ingestion de document            │
│  GET  /api/v1/health         → état de tous les services        │
│  GET  /api/v1/evaluation/…   → métriques d'évaluation           │
└─────┬──────────┬──────────┬──────────┬──────────────────────────┘
      │          │          │          │
┌─────▼──┐ ┌────▼───┐ ┌────▼───┐ ┌────▼────────────────────────┐
│ Milvus │ │  Elast │ │ Post-  │ │         Ollama               │
│ v2.5   │ │ search │ │ greSQL │ │  llama3.2:latest (LLM)       │
│ HNSW   │ │  8.13  │ │  15    │ │  mxbai-embed-large (embed)   │
│ 1024d  │ │  BM25  │ │  meta  │ └──────────────────────────────┘
└────────┘ └────────┘ │sessions│
                      │ audit  │ ┌────────────────────────────┐
                      └────────┘ │          Neo4j              │
                                 │  5.18 Community             │
                                 │  Graphe de connaissances    │
                                 │  (articles, concepts, refs) │
                                 └────────────────────────────┘
```

---

## Pipeline cognitif

Chaque requête traverse 7 étapes orchestrées par `CognitiveOrchestrator` :

| Étape | Nom | Description |
|-------|-----|-------------|
| 1 | **PERCEVOIR** | Normalisation de la requête, détection langue (FR/EN) |
| 2 | **COMPRENDRE** | Détection d'intention, extraction d'entités, système juridique, chargement historique de session depuis PostgreSQL |
| 3 | **DÉLIBÉRER** | Planification du plan d'action (HYBRID_SEARCH, RERANK, KNOWLEDGE_GRAPH, WEB_SEARCH_IF_LOW_CONFIDENCE…) |
| 4 | **AGIR** | Recherche hybride Milvus+ES, re-ranking, enrichissement Neo4j, fallback web si score < seuil |
| 5 | **GÉNÉRER** | Génération LLM (streaming ou batch) avec contexte, historique et avertissements |
| 6 | **VÉRIFIER** | Contrôle qualité : détection hors-domaine, longueur, citations |
| 7 | **ADAPTER** | Persistance PostgreSQL, mise à jour graphe, logs structurés |

---

## Stack technologique

| Composant | Technologie | Version | Rôle |
|-----------|-------------|---------|------|
| API | FastAPI + Uvicorn | 0.115 / 0.34 | Backend REST + SSE |
| LLM | Ollama / llama3.2 | latest | Génération de réponses |
| Embedding | Ollama / mxbai-embed-large | — | Vecteurs 1024d |
| Re-ranker | BAAI/bge-reranker-v2-m3 | HuggingFace | Cross-encoder |
| Vectordb | Milvus standalone | 2.5.6 | Recherche dense HNSW |
| Recherche lexicale | Elasticsearch | 8.13.4 | BM25 |
| Base relationnelle | PostgreSQL | 15 | Métadonnées, sessions, audit |
| Graphe | Neo4j Community | 5.18 | Connaissances juridiques |
| OCR | Tesseract 5 + PyMuPDF | — | Documents scannés |
| Conteneurs | Docker Compose | — | Déploiement on-premise |

---

## Structure du projet

```
modelAi/
├── app/
│   ├── api/
│   │   ├── middleware/         # Audit, rate-limiting
│   │   └── v1/
│   │       ├── health.py       # GET /api/v1/health
│   │       ├── ingest.py       # POST /api/v1/ingest
│   │       ├── query.py        # POST /api/v1/query[/stream]
│   │       └── evaluation.py   # GET /api/v1/evaluation/…
│   ├── core/
│   │   ├── config.py           # Settings Pydantic (env vars)
│   │   ├── dependencies.py     # Injection de dépendances FastAPI
│   │   ├── exceptions.py       # Gestionnaires d'erreurs globaux
│   │   └── logging.py          # Logging structuré JSON
│   ├── front/
│   │   ├── index.html          # SPA interface web
│   │   ├── app.js              # Logique JS (streaming, session, badges)
│   │   └── style.css           # Thème GOV-AI (bleu/vert CENADI)
│   ├── models/
│   │   ├── schemas.py          # Pydantic v2 — QueryRequest/Response, etc.
│   │   └── db_models.py        # SQLAlchemy — Document, Chunk, QueryLog…
│   ├── services/
│   │   ├── cognitive_orchestrator.py   # Pipeline 7 étapes
│   │   ├── embedding/          # mxbai-embed-large via Ollama
│   │   ├── ingestion/          # Extraction PDF/OCR + chunking + indexation
│   │   ├── reranking/          # BAAI/bge-reranker-v2-m3 (timeout 45s)
│   │   ├── knowledge_graph/    # Neo4j — entités et relations juridiques
│   │   ├── audit/              # Filtres sécurité, validation
│   │   └── evaluation/         # Métriques B0→B4 (MRR, nDCG, ISB…)
│   ├── storage/
│   │   ├── milvus_client.py    # Connexion + collection HNSW
│   │   ├── elasticsearch_client.py
│   │   ├── postgres_client.py
│   │   └── neo4j_client.py
│   └── main.py                 # Factory FastAPI + lifespan
├── eval/
│   ├── datasets/
│   │   └── qa_bilingual_annotated.json   # 30 requêtes annotées FR/EN
│   ├── run_eval.py             # Runner B0→B4
│   └── DATASET_REQUIREMENTS.md
├── Dockerfile                  # Python 3.11-slim + Tesseract + torch CPU
├── docker-compose.yml          # Stack complète 6 services
├── requirements.txt
└── .env.example                # Variables d'environnement documentées
```

---

## Démarrage rapide

### Prérequis

- Docker Desktop ≥ 4.30 (Linux containers)
- 16 Go RAM recommandés (Ollama + Milvus + Elasticsearch)
- 20 Go d'espace disque libre

### 1. Cloner et configurer

```bash
git clone <repo-url>
cd modelAi
cp .env.example .env
# Éditer .env si nécessaire (les valeurs par défaut fonctionnent en dev)
```

### 2. Construire l'image API

```bash
docker build -t govai2-api:2.0.0-sprint3 -t govai2-api:latest .
```

> La construction prend ~5 min (torch CPU ~800 MB + dépendances ML).

### 3. Démarrer la stack

```bash
docker compose up -d
```

Le démarrage complet prend ~2 min. Les modèles Ollama sont téléchargés automatiquement par `ollama-init` (~2.5 GB au total, une seule fois).

### 4. Vérifier la santé

```bash
curl http://localhost:8000/api/v1/health
```

Réponse attendue (tous les services `"ok"`) :

```json
{
  "status": "healthy",
  "version": "2.0.0-sprint3",
  "services": {
    "milvus": "ok",
    "elasticsearch": "ok",
    "postgres": "ok",
    "neo4j": "ok",
    "ollama": "ok",
    "reranker": "ok"
  }
}
```

### 5. Accéder à l'interface

| URL | Description |
|-----|-------------|
| `http://localhost:8000/ui` | Interface web GOV-AI 2.0 |
| `http://localhost:8000/docs` | Swagger UI (API interactive) |
| `http://localhost:8000/redoc` | ReDoc |
| `http://localhost:7474` | Neo4j Browser |

---

## API — Endpoints principaux

### POST `/api/v1/query`

Soumet une requête juridique au pipeline cognitif complet.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Quelles sont les sanctions pour corruption selon le Code pénal ?",
    "session_id": "session-demo-001",
    "top_k": 5,
    "juridical_system": "civil_law"
  }'
```

**Corps de la réponse :**

```json
{
  "answer": "Selon l'article 134 du Code pénal camerounais...",
  "citations": [
    {
      "doc_title": "Code Pénal Camerounais",
      "article": "Article 134",
      "jurisdiction": "national",
      "relevance_score": 0.92,
      "chunk_text": "..."
    }
  ],
  "intent_detected": "legal_information",
  "juridical_system_detected": "civil_law",
  "action_plan": ["HYBRID_SEARCH", "RERANK", "KNOWLEDGE_GRAPH"],
  "session_id": "session-demo-001",
  "language": "fr",
  "latency_ms": 4200
}
```

### POST `/api/v1/query/stream`

Même paramètres que `/query`, réponse en **Server-Sent Events** (token par token).

```bash
curl -N -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "How to register a company in Cameroon?", "session_id": "s1"}'
```

### GET `/api/v1/health`

État détaillé de tous les services (Milvus, ES, PostgreSQL, Neo4j, Ollama, reranker).

---

## Ingestion de documents

### Via l'API (multipart/form-data)

```bash
# Ingérer le Code pénal camerounais
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@Cameroon - Penal Code.pdf" \
  -F "doc_type=code_penal" \
  -F "institution=Ministère de la Justice" \
  -F "jurisdiction=national"
```

**Réponse :**

```json
{
  "document_id": "doc-uuid-...",
  "filename": "Cameroon - Penal Code.pdf",
  "chunks_created": 342,
  "language_detected": "en",
  "ocr_used": false,
  "status": "success"
}
```

### Pipeline d'ingestion

1. Validation sécurité (nom de fichier, taille max 50 MB, extension)
2. Extraction texte : PyMuPDF → OCR Tesseract si texte < seuil
3. Chunking structurel (articles, sections) avec overlap configurable
4. Embedding via Ollama/mxbai-embed-large (1024d)
5. Indexation Milvus (dense HNSW) + Elasticsearch (BM25)
6. Persistance métadonnées PostgreSQL
7. Extraction entités → Neo4j (concepts, références croisées)

### Supprimer un document

```bash
curl -X DELETE http://localhost:8000/api/v1/ingest/{doc_id}
```

---

## Interface web

L'interface `/ui` est une SPA vanilla JS qui expose :

- **Zone de chat** — questions en FR ou EN, réponses avec citations cliquables
- **Badges de pipeline** — `KG` (vert, Knowledge Graph utilisé) · `Web` (bleu, recherche web) · `🔒 Souverain`
- **Session persistante** — icône base de données quand l'historique PostgreSQL est actif
- **Mode streaming** — affichage token par token, badge ⚡
- **Avertissements** — bloc orange si hors-domaine, bloc bleu si résultats web inclus
- **Citations** — score de pertinence, titre, article, juridiction

---

## Évaluation

L'étude ablative B0→B4 mesure l'apport de chaque composant du pipeline :

| Baseline | Description |
|----------|-------------|
| B0 | Recherche dense seule (Milvus HNSW) |
| B1 | B0 + BM25 Elasticsearch (hybride) |
| B2 | B1 + re-ranking (BAAI/bge-reranker-v2-m3) |
| B3 | B2 + graphe de connaissances Neo4j |
| B4 | B3 + historique de session + tout le pipeline Sprint 3 |

**Métriques :**

- `P@k`, `nDCG@k`, `MRR` — qualité du retrieval
- `Faithfulness`, `Citation Precision` (cible ≥ 0.95), `Hallucination Rate` (cible ≤ 0.05)
- `ISB` (Indice de Souveraineté et de Biais) — cible ≥ 0.85
- `p50` / `p95` latence — cibles ≤ 5 000 ms / 15 000 ms

**Lancer une évaluation :**

```bash
# Depuis le conteneur API
docker exec govai2-api python eval/run_eval.py --baseline B4 --dataset eval/datasets/qa_bilingual_annotated.json --k 1 3 5 10
```

---

## Configuration

Toutes les variables d'environnement sont documentées dans `.env.example`. Les principales :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `LLM_MODEL` | `llama3.2:latest` | Modèle LLM Ollama |
| `EMBEDDING_MODEL` | `mxbai-embed-large` | Modèle d'embedding |
| `EMBEDDING_DIM` | `1024` | Dimension des vecteurs |
| `SESSION_HISTORY_MAX_TURNS` | `5` | Tours d'historique chargés depuis PostgreSQL |
| `WEB_SEARCH_ENABLED` | `false` | Activer le fallback DuckDuckGo (opt-in) |
| `WEB_SEARCH_MAX_RESULTS` | `3` | Résultats web maximum |
| `NEO4J_URI` | `bolt://neo4j:7687` | Connexion Neo4j |
| `LOG_LEVEL` | `INFO` | Niveau de log (DEBUG / INFO / WARNING) |
| `LOG_FORMAT` | `json` | Format des logs (json / text) |

---

## Développement local

### Hot-reload (sans rebuild Docker)

Le `docker-compose.yml` monte le code source en bind-mount (`- .:/app`) et active `--reload`. Toute modification de fichier `.py` relance Uvicorn automatiquement.

```bash
# Voir les logs en temps réel
docker compose logs -f govai-api

# Redémarrer l'API après un changement qui bloque le reload
docker restart govai2-api
```

### Accès aux bases de données

```bash
# PostgreSQL
docker exec -it govai2-postgres psql -U govai -d govai2

# Neo4j Browser
# Ouvrir http://localhost:7474 — login : neo4j / neo4j_secret

# Elasticsearch
curl http://localhost:9200/_cat/indices?v

# Milvus (via pymilvus en Python)
docker exec -it govai2-api python -c "
from pymilvus import connections, utility
connections.connect(host='milvus', port=19530)
print(utility.list_collections())
"
```

### Pré-chauffer Ollama (premier démarrage)

```bash
docker exec govai2-ollama ollama run llama3.2 "Bonjour" --nowordwrap
```

---

## Sécurité

- Utilisateur non-root `govai` dans le conteneur
- Validation des noms de fichier à l'ingestion (path traversal, extensions)
- Rate limiting par IP (middleware)
- Audit log structuré de toutes les requêtes (PostgreSQL)
- CORS restreint aux domaines CENADI en production (`APP_ENV=production`)
- Aucune donnée vers l'extérieur (LLM, embedding, graphe — 100 % on-premise)

---

## Contributeurs

- **NJILLA TCHAGADICK NICOL EMMANUEL** — Master II Informatique, ENSPY Yaoundé

---

## Licence

Propriétaire — CENADI / ENSPY. Tous droits réservés.
