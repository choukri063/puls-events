# 🎭 Puls-Events — POC Système RAG

Assistant culturel intelligent basé sur un système **Retrieval-Augmented Generation (RAG)**,
combinant **Mistral AI**, **FAISS** et **LangChain** pour recommander des événements culturels
en **Île-de-France** et **Hauts-de-France**.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue?style=flat-square)
![Mistral AI](https://img.shields.io/badge/Mistral_AI-LLM-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)

---

# 📌 Présentation du projet

Ce POC (**Proof of Concept**) a été développé pour **Puls-Events**, une plateforme de découverte
d'événements culturels.

L'objectif est de démontrer la faisabilité d'un assistant conversationnel capable de :

- 🎯 Recommander des événements culturels personnalisés
- 💬 Répondre à des questions en langage naturel
- 🔎 Effectuer une recherche sémantique dans une base vectorielle FAISS
- 🤖 Générer des réponses contextualisées avec un LLM Mistral

Les données utilisées proviennent d'**OpenAgenda**.

---

# 🏗️ Architecture RAG

```
Utilisateur
    │
    ▼
Question en langage naturel
    │
    ▼
Embedding Mistral (mistral-embed)
    │
    ▼
Recherche vectorielle FAISS
    │
    ▼
Top-K documents similaires
    │
    ▼
Construction du contexte événementiel
    │
    ▼
Mistral LLM (mistral-small-latest)
    │
    ▼
Réponse générée
    │
    ▼
Utilisateur
```

---

# 📂 Structure du projet

```
P11_AHMED_CHOUKRI_Concevez_et_deployez_un_systeme_RAG/

├── .env
├── requirements.txt
├── README.md

├── scripts/
│   ├── 01_preprocessing.py
│   └── 02_vectorisation.py

├── rag/
│   ├── rag_chatbot.py
│   └── demo_chatbot.py

├── app/
│   └── app.py

├── tests/
│   └── test_preprocessing.py

└── generated_data/
    ├── data_processed.json
    ├── faiss_index.bin
    └── faiss_metadata.pkl
```

---

# ⚙️ Installation

## Prérequis

- Python 3.10+
- Une clé API Mistral

---

## 1. Cloner le projet

```bash
git clone <url-du-repo>

cd P11_AHMED_CHOUKRI_Concevez_et_deployez_un_systeme_RAG
```

---

## 2. Créer l'environnement virtuel

```bash
python -m venv venv
```

Activation :

Windows :

```bash
venv\Scripts\activate
```

Linux/macOS :

```bash
source venv/bin/activate
```

---

## 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 4. Variables d'environnement

Créer un fichier `.env`

```env
MISTRAL_API_KEY=votre_clé_api
```

---

# 🚀 Utilisation

## 1) Préprocessing

```bash
python 01_preprocessing.py
```

Traitements réalisés :

- Nettoyage des données CSV
- Filtrage géographique :
  - Île-de-France
  - Hauts-de-France
- Filtrage temporel (< 1 an)

Résultat :

```
data_processed.json
```

---

## 2) Vectorisation FAISS

```bash
python 02_vectorisation.py
```

Création :

```
faiss_index.bin
faiss_metadata.pkl
```

Process :

- Découpage en chunks
- Génération embeddings Mistral
- Indexation vectorielle FAISS

---

## 3) Chatbot CLI

```bash
python demo_chatbot.py
```

Commandes :

```
sources  → afficher les sources
quitter  → fermer
```

---

## 4) Application Streamlit

```bash
streamlit run app.py
```

Disponible :

```
http://localhost:8501
```

---

# 🧪 Tests

```bash
pytest test_preprocessing.py -v
```

Tests réalisés :

✅ JSON valide  
✅ Champs obligatoires présents  
✅ Régions autorisées  
✅ Dates valides  
✅ Texte documentaire généré  
✅ Absence de doublons  
✅ Volume cohérent  

Résultat :

```
8/8 tests réussis
```

---

# 🛠️ Stack technique

| Composant | Technologie |
|-|-|
| Langage | Python |
| LLM | Mistral AI |
| Embeddings | mistral-embed |
| Vector Database | FAISS |
| Framework RAG | LangChain |
| Interface | Streamlit |
| Data Processing | Pandas |
| Tests | Pytest |
| Dataset | OpenAgenda |

---

# 📊 Données traitées

| Indicateur | Valeur |
|-|-:|
| Événements bruts | 112 701 |
| Après filtre géographique | 28 618 |
| Après filtre temporel | 3 879 |
| Chunks vectorisés | 4 100 |
| Dimension embeddings | 1 024 |
| Régions | IDF + Hauts-de-France |

---

# 🔒 Fichiers exclus Git

```
.env
faiss_index.bin
faiss_metadata.pkl
data_processed.json
*.csv
*.xlsx
__pycache__/
venv/
```

---

# 👤 Auteur

**Ahmed Choukri**  
Ingénieur Data Freelance

Spécialisation :

- NLP
- RAG
- Bases vectorielles
- Data Engineering
- IA générative

---

Projet réalisé dans le cadre d'un POC RAG pour **Puls-Events**.
