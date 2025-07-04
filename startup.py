#!/usr/bin/env python3
"""
Script de démarrage pour l'API du système RAG juridique camerounais.
Ce script initialise l'environnement et démarre l'API.
"""

import os
import sys
import logging
import subprocess
import argparse
from fastapi import FastAPI

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Démarrage de l'API RAG Juridique")
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Adresse d'hôte pour l'API"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port pour l'API"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Activer le rechargement automatique"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Activer le mode debug"
    )
    
    return parser.parse_args()

def check_environment():
    """Vérifie que l'environnement est correctement configuré."""
    # Vérifier les répertoires de données
    data_dirs = [
        "/Users/imacpro/modelAi/data",
        "/Users/imacpro/modelAi/metadata",
        "/Users/imacpro/modelAi/metadata/chunks"
    ]
    
    for directory in data_dirs:
        if not os.path.exists(directory):
            logger.info(f"Création du répertoire {directory}")
            os.makedirs(directory, exist_ok=True)
   
    ''' # Vérifier la connexion SSH
    try:
        # Test simple de connexion SSH (remplacer par un test plus robuste si nécessaire)
        result = subprocess.run(
            ["ssh", "-q", "-o", "BatchMode=yes", "imacpro@10.100.212.118", "echo OK"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0 or "OK" not in result.stdout:
            logger.warning("⚠️ La connexion SSH à imacpro@10.100.212.118 semble ne pas fonctionner correctement.")
            logger.warning("Vérifiez que la clé SSH est configurée et que le serveur est accessible.")
        else:
            logger.info("✅ Connexion SSH à imacpro@10.100.212.118 OK")
    except Exception as e:
        logger.warning(f"⚠️ Erreur lors du test de connexion SSH: {e}") '''

def main():
    """Fonction principale pour démarrer l'API."""
    args = parse_args()
    logger.info("🚀 Démarrage du système RAG juridique camerounais")
    
    # IMPORTANT: Définir les variables d'environnement avant de démarrer l'API
    os.environ["MILVUS_HOST"] = "10.100.212.133"
    os.environ["MILVUS_PORT"] = "19530"
    logger.info(f"Variables d'environnement Milvus définies: host={os.environ['MILVUS_HOST']}, port={os.environ['MILVUS_PORT']}")
    
    # Vérifier l'environnement
    check_environment()
    
    # Préparer la commande uvicorn
    cmd = [
        "uvicorn",
        "app.main:app",
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    if args.reload:
        cmd.append("--reload")
    
    if args.debug:
        cmd.append("--log-level=debug")
    else:
        cmd.append("--log-level=info")
    
    # Démarrer l'API
    logger.info(f"🌐 Démarrage de l'API sur {args.host}:{args.port}")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("👋 Arrêt de l'API")
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage de l'API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()