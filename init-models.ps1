# ============================================
# Script d'initialisation des modeles Ollama
# pour GOV-AI (a executer apres docker-compose up)
# ============================================

Write-Host "=== Initialisation des modeles GOV-AI ===" -ForegroundColor Cyan

Write-Host "`n[1/2] Telechargement du modele d'embedding (mxbai-embed-large)..." -ForegroundColor Yellow
docker exec ollama ollama pull mxbai-embed-large

Write-Host "`n[2/2] Telechargement du modele LLM (llama3.2)..." -ForegroundColor Yellow
docker exec ollama ollama pull llama3.2

Write-Host "`n=== Verification des modeles ===" -ForegroundColor Cyan
docker exec ollama ollama list

Write-Host "`n=== Modeles prets ! ===" -ForegroundColor Green
Write-Host "Vous pouvez maintenant lancer: python startup.py" -ForegroundColor Green
