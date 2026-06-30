Puls-Events — POC Système RAG

Assistant culturel intelligent basé sur un système de génération augmentée par récupération (RAG),
combinant **Mistral AI**, **FAISS** et **LangChain** pour recommander des événements culturels
en **Île-de-France** et **Hauts-de-France**.

---

## Présentation du projet

Ce POC (Proof of Concept) a été développé pour **Puls-Events**, plateforme de découverte
d'événements culturels. Il démontre la faisabilité d'un chatbot intelligent capable de :

- Recommander des événements culturels personnalisés
- Répondre en langage naturel à des questions sur les événements
- Effectuer des recherches sémantiques dans une base vectorielle FAISS
- S'appuyer sur les données OpenAgenda (événements < 1 an)

---

## Architecture
Utilisateur
│
▼
Question (langage naturel)
│
▼
Mistral Embeddings  ──►  Vecteur requête
│
▼
Recherche FAISS  ──►  Top-K chunks similaires
│
▼
Construction du contexte (métadonnées événements)
│
▼
Mistral LLM (mistral-small-latest)  ──►  Réponse naturelle
│
▼
Utilisateur

---

##  Structure du projet
P11_AHMED_CHOUKRI_Concevez_et_deployez_un_systeme_RAG/
│
├── 📄 .env                          # Clés API (non versionné)
├── 📄 requirements.txt              # Dépendances Python
├── 📄 README.md                     # Ce fichier
│
├──  Scripts de traitement
│   ├── 01_preprocessing.py          # Chargement, filtrage, nettoyage CSV
│   └── 02_vectorisation.py          # Embeddings Mistral + Index FAISS
│
├──  Système RAG
│   ├── rag_chatbot.py               # Moteur RAG (embed + recherche + LLM)
│   └── demo_chatbot.py              # Démo interactive CLI
│
├──  Interface web
│   └── app.py                       # Interface Streamlit
│
├──  Tests
│   └── test_preprocessing.py        # Tests unitaires (pytest)
│
└──  Données générées
├── data_processed.json          # Événements filtrés et nettoyés
├── faiss_index.bin              # Index vectoriel FAISS
└── faiss_metadata.pkl           # Métadonnées des chunks

---

##️ Installation

### Prérequis

- Python 3.10+
- Clé API Mistral ([console.mistral.ai](https://console.mistral.ai))

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd P11_AHMED_CHOUKRI_Concevez_et_deployez_un_systeme_RAG
```

### 2. Créer et activer un environnement virtuel

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

Créer un fichier `.env` à la racine :

```env
MISTRAL_API_KEY=votre_clé_api_mistral
```

### 5. Vérifier l'environnement

```bash
python check_env.py
```

---

## Utilisation

### Étape 1 — Preprocessing des données

```bash
python 01_preprocessing.py
```

Filtre les événements OpenAgenda :
- Régions : Île-de-France + Hauts-de-France
- Période : moins d'un an
- Génère : `data_processed.json` (3 879 événements)

### Étape 2 — Vectorisation et indexation FAISS

```bash
python 02_vectorisation.py
```

- Découpe les textes en chunks
- Génère les embeddings via Mistral (`mistral-embed`)
- Construit l'index FAISS
- Génère : `faiss_index.bin` + `faiss_metadata.pkl` (4 100 chunks)

### Étape 3 — Démo CLI

```bash
python demo_chatbot.py
```

Interface en ligne de commande pour tester le chatbot.
Commandes disponibles : `sources` (voir les sources), `quitter` (exit)

### Étape 4 — Interface web Streamlit

```bash
python -m streamlit run app.py
```

Interface graphique accessible sur `http://localhost:8501`

---

## Tests unitaires

```bash
pytest test_preprocessing.py -v
```

**8 tests couvrant :**
- Fichier JSON non vide
- Présence de toutes les clés attendues
- Régions valides (Île-de-France / Hauts-de-France uniquement)
- Dates < 1 an
- Titre ou description non vide
- `texte_document` non vide
- Absence de doublons
- Volume minimum cohérent (≥ 100 événements)

---

## Stack technique

| Composant | Technologie | Version |
|---|---|---|
| Langage | Python | 3.14 |
| LLM | Mistral AI (`mistral-small-latest`) | 2.4.7 |
| Embeddings | Mistral AI (`mistral-embed`) | 2.4.7 |
| Base vectorielle | FAISS | 1.13.2 |
| Orchestration | LangChain | 0.4.1 |
| Interface web | Streamlit | latest |
| Manipulation données | Pandas | 2.3.3 |
| Tests | Pytest | 8.4.2 |
| Source données | OpenAgenda | — |

---

##  Données

| Indicateur | Valeur |
|---|---|
| Source | OpenAgenda (export CSV) |
| Événements bruts | 112 701 |
| Après filtre géographique | 28 618 |
| Après filtre temporel (< 1 an) | 3 879 |
| Chunks vectorisés | 4 100 |
| Dimension des vecteurs | 1 024 |
| Régions couvertes | Île-de-France, Hauts-de-France |

---

## Fichiers non versionnés

Les fichiers suivants sont exclus du dépôt git (`.gitignore`) :
.env
faiss_index.bin
faiss_metadata.pkl
data_processed.json
evenements-publics-openagenda.csv
evenements-publics-openagenda.json
evenements-publics-openagenda.xlsx
pycache/
venv/

---

## Auteur

**Ahmed Choukri** — Ingénieur Data Freelance  
Spécialisation : NLP, bases de données vectorielles, systèmes RAG

---

*Projet réalisé pour Puls-Events dans le cadre d'un POC RAG.*
