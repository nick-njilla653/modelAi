# Interface GOV-AI 2.0

Interface conversationnelle en React + Vite + TypeScript.

## Pourquoi la construction se fait sur l'hôte

L'image applicative n'embarque pas Node : elle sert Python et rien d'autre.
Le build se fait donc sur la machine de développement, et FastAPI sert le
répertoire `dist/` produit — visible dans le conteneur par le bind-mount du
dépôt.

## Développement

```bash
cd frontend
npm install
npm run dev
```

L'interface écoute sur <http://localhost:5173> et relaie `/api` vers FastAPI sur
`:8000` (voir `vite.config.ts`). Aucune configuration CORS n'est nécessaire.

L'API doit tourner en parallèle :

```bash
docker compose up -d
```

## Production

```bash
npm run build
```

Produit `frontend/dist/`, monté automatiquement par FastAPI sous `/ui` au
démarrage. Sans build disponible, `/ui` sert l'ancienne interface plutôt qu'une
erreur — et le démarrage journalise `webui_build_missing`.

L'ancienne interface reste accessible sous `/ui/legacy` : elle porte encore les
onglets Ingestion, Évaluation et Fine-Tuning, non repris ici.

## Ce que l'interface expose

| Élément | Route consommée |
|---|---|
| Liste des conversations | `GET /api/v1/sessions` |
| Réouverture d'une conversation | `GET /api/v1/sessions/{id}` |
| Suppression | `DELETE /api/v1/sessions/{id}` |
| Inventaire du corpus | `GET /api/v1/documents` |
| Question (flux) | `POST /api/v1/query/stream` |

Le flux SSE émet des événements typés, dans l'ordre :

```
stage  → jalon de progression (analyse, recherche, rédaction)
meta   → intention détectée, plan d'action, nombre d'extraits
token  → fragments de texte
done   → citations résolues, drapeaux de sûreté, latence
```

Les jalons `stage` ne sont pas cosmétiques : sur le matériel de développement,
la recherche et le reclassement précèdent le premier token de près de deux
minutes. Sans eux, l'interface paraît figée et l'utilisateur relance sa requête.

## Contrat de types

`src/types.ts` reproduit à la main les schémas Pydantic de
`app/models/schemas.py`. Toute évolution de `QueryResponse`, `Citation` ou
`SessionTurn` doit y être répercutée — `npm run typecheck` ne détectera pas une
divergence avec le serveur, seulement une incohérence interne.
