"""
GOV-AI 2.0 — Client Neo4j async (Sprint 2).
Graphe de connaissances juridiques camerounais.

Schéma ontologique :
  (TexteNormatif)-[:A_ARTICLE]->(Article)
  (Article)-[:REFERENCE]->(Article)
  (Article)-[:ABROGE]->(Article)
  (Article)-[:MODIFIE]->(Article)
  (Concept)-[:DEFINI_DANS]->(Article)
  (Institution)-[:EMETRICE_DE]->(TexteNormatif)
"""
from __future__ import annotations

from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_driver: Any = None


def get_neo4j_driver() -> Any:
    """Retourne le driver Neo4j (singleton)."""
    global _driver
    if _driver is not None:
        return _driver

    settings = get_settings()
    try:
        from neo4j import AsyncGraphDatabase
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            database=settings.neo4j_database,
            max_connection_pool_size=10,
        )
        logger.info("neo4j_driver_created", uri=settings.neo4j_uri)
    except ImportError:
        logger.warning("neo4j_package_missing", hint="pip install neo4j")
        raise
    except Exception as exc:
        logger.error("neo4j_driver_failed", error=str(exc))
        raise

    return _driver


async def close_neo4j() -> None:
    """Ferme le driver Neo4j."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
        logger.info("neo4j_driver_closed")


async def check_neo4j_connection() -> bool:
    """Vérifie la connexion Neo4j."""
    try:
        driver = get_neo4j_driver()
        async with driver.session() as session:
            await session.run("RETURN 1")
        return True
    except Exception as exc:
        logger.error("neo4j_connection_failed", error=str(exc))
        return False


async def run_query(
    cypher: str,
    params: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Exécute une requête Cypher et retourne les résultats."""
    driver = get_neo4j_driver()
    async with driver.session() as session:
        result = await session.run(cypher, params or {})
        return [record.data() async for record in result]


async def ensure_schema() -> None:
    """Crée les index et contraintes Neo4j si absents."""
    constraints = [
        "CREATE CONSTRAINT texte_id IF NOT EXISTS FOR (t:TexteNormatif) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT institution_id IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE",
    ]
    indexes = [
        "CREATE INDEX texte_title IF NOT EXISTS FOR (t:TexteNormatif) ON (t.title)",
        "CREATE INDEX article_number IF NOT EXISTS FOR (a:Article) ON (a.number)",
        "CREATE INDEX concept_label IF NOT EXISTS FOR (c:Concept) ON (c.label)",
    ]
    driver = get_neo4j_driver()
    async with driver.session() as session:
        for stmt in constraints + indexes:
            try:
                await session.run(stmt)
            except Exception as exc:
                logger.debug("neo4j_schema_stmt_skipped", stmt=stmt[:60], error=str(exc))
    logger.info("neo4j_schema_ready")
