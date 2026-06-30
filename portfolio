# 👋 Ahmed Choukri — Data Engineer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.4-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![Mistral](https://img.shields.io/badge/Mistral_AI-orange?style=for-the-badge&logo=ai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-1.13-blue?style=for-the-badge)
![GCP](https://img.shields.io/badge/GCP-Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**Spécialisation : NLP · Bases de données vectorielles · Systèmes RAG · Data Engineering**

*Data Engineer en alternance — Passionné par les LLM et l'IA appliquée*

</div>

---

## 🚀 Projet Phare

### 🎭 [Puls-Events — Système RAG pour la Recommandation d'Événements Culturels](https://github.com/ahmed-choukri/puls-events-rag)

> Assistant culturel intelligent basé sur un système **RAG (Retrieval-Augmented Generation)**
> combinant **Mistral AI**, **FAISS** et **LangChain** pour recommander des événements culturels
> en **Île-de-France** et **Hauts-de-France**.

#### 🏗️ Architecture du système

```
Question utilisateur (langage naturel)
         │
         ▼
Mistral Embeddings (1024 dim)   ◄── LangChain MistralAIEmbeddings
         │
         ▼
Recherche FAISS IndexIDMap (< 2ms)
         │
         ▼
Contexte événements (LangChain Documents)
         │
         ▼
ChatPromptTemplate + mistral-small-latest
         │
         ▼
StrOutputParser → Réponse naturelle en français
```

#### 📊 Résultats mesurés

| Indicateur | Valeur |
|---|---|
| Événements indexés | **3 879** |
| Chunks vectorisés | **4 100** |
| Dimension vecteurs | **1 024** (mistral-embed) |
| Temps réponse moyen | **2 003 ms** |
| Recherche FAISS | **2 ms** (< 0.1% du temps total) |
| Tests unitaires | **8/8 ✅** |
| Paires Q/R annotées | **50** |

#### 🔧 Stack technique

```
LLM          : Mistral AI (mistral-small-latest)
Embeddings   : mistral-embed (1024 dimensions)
Vectoriel    : FAISS IndexIDMap + distance L2
Orchestration: LangChain 0.4 (ChatPromptTemplate, MistralAIEmbeddings, StrOutputParser)
Interface    : Streamlit + CLI interactive
Données      : OpenAgenda (112 701 événements bruts → 3 879 filtrés)
Tests        : Pytest 8.4 (8/8 passés en 1.15s)
```

#### 🚀 Lancer le projet

```bash
git clone https://github.com/ahmed-choukri/puls-events-rag
cd puls-events-rag
pip install -r requirements.txt
cp .env.example .env   # Ajouter MISTRAL_API_KEY

python scripts/01_preprocessing.py    # ~30 secondes
python scripts/02_vectorisation.py    # ~8 minutes (appels API)
python -m streamlit run app/app.py    # Interface web
```

#### 📈 Benchmark performances réels

```
Embedding Mistral  :  320 ms  ████████░░░░░░░░  16%
Recherche FAISS    :    2 ms  ░░░░░░░░░░░░░░░░  < 1%
Génération LLM     : 1680 ms  ████████████████  84%
─────────────────────────────────────────────────
TOTAL moyen        : 2003 ms  (~2 secondes)
```

---

## 🛠️ Compétences techniques

### NLP & Intelligence Artificielle
```
RAG (Retrieval-Augmented Generation)    ████████████████  ★★★★★
LangChain (chains, prompts, memory)     ████████████████  ★★★★★
Mistral AI (LLM + Embeddings)           ████████████████  ★★★★★
Prompt Engineering                      ███████████████░  ★★★★☆
Fine-tuning (Vertex AI)                 █████████░░░░░░░  ★★★☆☆
```

### Bases de données vectorielles
```
FAISS (IndexFlatL2, IndexIDMap)         ████████████████  ★★★★★
Qdrant (filtres metadata, Cloud)        ████████████░░░░  ★★★★☆
Chunking sémantique                     ████████████████  ★★★★★
Indexation vectorielle                  ███████████████░  ★★★★☆
```

### Data Engineering
```
Python (Pandas, NumPy, tqdm)            ████████████████  ★★★★★
ETL Pipeline (CSV → JSON → Vecteurs)    ████████████████  ★★★★★
Pytest (tests unitaires)                ███████████████░  ★★★★☆
API REST (FastAPI)                      ████████████░░░░  ★★★★☆
```

### Cloud & DevOps
```
GCP (Cloud Run, Cloud Functions, GCS)   ████████████░░░░  ★★★★☆
Docker & CI/CD                          ███████████░░░░░  ★★★☆☆
LangSmith (monitoring LLM)              ████████████░░░░  ★★★★☆
Streamlit (déploiement)                 ████████████████  ★★★★★
```

---

## 📦 Projets

### 🎭 POC RAG — Puls-Events
**Système de recommandation d'événements culturels par IA**

- Pipeline complet : OpenAgenda CSV → Preprocessing → FAISS → RAG → Streamlit
- LangChain + Mistral AI + FAISS + Pytest
- **3 879 événements**, **2 secondes** de réponse, **8/8 tests**

[![GitHub](https://img.shields.io/badge/GitHub-View_Project-181717?style=flat-square&logo=github)](https://github.com/choukri063/puls-events-rag)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-blue?style=flat-square)
![Mistral](https://img.shields.io/badge/Mistral_AI-orange?style=flat-square)

---

### 🗺️ MVP Design — Puls-Events (POC → MVP)
**Étude de design pour la mise en production du système RAG**

- Architecture cloud GCP (Cloud Run + Qdrant + Redis + FastAPI)
- Nouvelles fonctionnalités : mémoire conversationnelle, géolocalisation, smolagents, monitoring
- **16 200€** Build · **95 à 2 220€/mois** OPEX · Planning **12 semaines**

[![GitHub](https://img.shields.io/badge/GitHub-View_Project-181717?style=flat-square&logo=github)](https://github.com/choukri063/puls-events-mvp)
![GCP](https://img.shields.io/badge/GCP-4285F4?style=flat-square&logo=google-cloud&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=flat-square)

---

## 🎓 Formation & Certifications

| Formation | Établissement | Année |
|---|---|---|
| Data Engineer (alternance) | OpenClassrooms | 2025-2026 |
| Spécialisation NLP & Bases Vectorielles | Formation continue | 2026 |

---

## 📫 Contact

<div align="center">

<a href="https://www.linkedin.com/in/ahmed-choukri-3b163a99/">
<img src="https://img.shields.io/badge/LinkedIn-Ahmed_Choukri-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
</a>

<a href="https://github.com/choukri063/data-engineer-portfolio">
<img src="https://img.shields.io/badge/GitHub-choukri063-181717?style=for-the-badge&logo=github&logoColor=white"/>
</a>

<a href="mailto:ahmedchoukri3@gmail.com">
<img src="https://img.shields.io/badge/Email-ahmedchoukri3%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white"/>
</a>

<br>

<i>Disponible pour des missions freelance en NLP, RAG et Data Engineering</i>

</div>
---

<div align="center">

*"Transformer la donnée brute en expérience utilisateur intelligente."*
![Profile Views](https://komarev.com/ghpvc/?username=ahmed-choukri&color=E94560&style=flat-square)
</div>
