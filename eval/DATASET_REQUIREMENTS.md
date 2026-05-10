# Demande de Livrables Documentaires — GOV-AI 2.0

## 1) Objectif

Ce document précise **ce que l’équipe doit fournir**, dans des formats simples :
- documents texte,
- fichiers Word,
- PDF,
- PDF scannés.

Important : **vous ne devez pas faire de préparation technique** (JSON, CSV, balisage, etc.).  
L’équipe technique se charge ensuite de l’import, du nettoyage et de l’adaptation pour les tests/évaluations.

## 2) Formats acceptés

- `.txt`
- `.doc` / `.docx`
- `.pdf` (texte natif)
- `.pdf` scanné (image)

## 3) Ce que vous devez nous livrer

### A. Corpus juridique principal (obligatoire)

Fournir les textes juridiques de référence (FR et EN si disponible), par exemple :
- codes (civil, pénal, fiscal, travail, etc.),
- lois, décrets, arrêtés,
- textes OHADA,
- guides administratifs officiels.

Attendus :
- versions les plus fiables disponibles,
- documents complets (pas d’extraits isolés si possible),
- scans lisibles si document non numérique.

---

### B. Documents pour cas pratiques / procédures (obligatoire)

Fournir les documents décrivant les procédures administratives et pratiques terrain :
- obtention acte de naissance,
- création d’entreprise,
- permis de construire,
- contentieux administratif,
- autres procédures fréquentes.

Attendus :
- documents réellement utilisés par usagers/agents,
- contenu suffisamment détaillé (étapes, délais, pièces).

---

### C. Documents bilingues FR/EN (très important)

Fournir, quand possible, les versions françaises et anglaises d’un même sujet juridique, pour vérifier l’équilibre bilingue.

Attendus :
- paires FR/EN sur les mêmes thèmes,
- signaler clairement quand seule une langue existe.

---

### D. Documents “hors périmètre” (pour tests de refus)

Fournir un petit lot de documents non juridiques ou non administratifs (sport, météo, divertissement, etc.) pour tester que le système refuse correctement.

Volume recommandé :
- 20 à 50 documents/cas.

---

### E. Documents sensibles aux tentatives de manipulation (sécurité)

Fournir des exemples de contenu contenant :
- formulations trompeuses,
- consignes contradictoires,
- tentatives de détourner la demande.

Objectif : tester la robustesse/sécurité du système.

Volume recommandé :
- 30 à 60 cas.

## 4) Informations minimales à joindre pour chaque fichier

Pour chaque document livré, merci d’ajouter (dans le nom du fichier ou dans un document d’accompagnement Word/PDF) :
- titre du document,
- langue (`FR` ou `EN`),
- type (code, loi, décret, guide, autre),
- source (institution / provenance),
- date du document (si connue),
- remarque utile (ex: “scan ancien”, “version partielle”, “traduction non officielle”).

## 5) Règles de nommage simples (recommandées)

Format conseillé :
`Type_Titre_Langue_Annee.ext`

Exemples :
- `Code_Penal_Cameroun_FR_2016.pdf`
- `Guide_Creation_Entreprise_FR_2023.docx`
- `OHADA_AUSCGIE_EN_2014.pdf`

## 6) Qualité attendue des documents

- Fichiers lisibles et ouvrables.
- Pages non coupées et dans le bon ordre.
- Scans nets (éviter flou, pages inclinées, texte illisible).
- Pas de doublons inutiles.

## 7) Organisation conseillée du dépôt de livraison

Vous pouvez simplement organiser vos fichiers ainsi :

```text
Livraison_Dataset/
  01_Corpus_Juridique/
  02_Procedures_Pratiques/
  03_Bilingue_FR_EN/
  04_Hors_Perimetre/
  05_Securite_Manipulation/
  06_Annexes/
```

## 8) Checklist rapide (équipe de collecte)

- [ ] Les fichiers sont en `.txt`, `.doc/.docx`, `.pdf` ou PDF scanné.
- [ ] Le corpus juridique principal est livré.
- [ ] Les documents de procédures pratiques sont livrés.
- [ ] Un lot bilingue FR/EN est livré (quand disponible).
- [ ] Un lot hors périmètre est livré.
- [ ] Un lot sécurité/manipulation est livré.
- [ ] Chaque fichier a au minimum titre, langue, source, date (si connue).
- [ ] Les scans sont lisibles.

## 9) Message clé pour l’équipe

Votre mission est de **fournir les meilleurs documents bruts possibles**.  
La transformation technique en datasets de test et d’évaluation sera entièrement prise en charge par notre équipe.

