"""
rag_chatbot.py
Puls-Events — Moteur RAG avec LangChain + Mistral + FAISS
"""

import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

load_dotenv()

MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY")
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH    = "faiss_metadata.pkl"
EMBED_MODEL      = "mistral-embed"
CHAT_MODEL       = "mistral-small-latest"
TOP_K            = 5

# ── Prompt système via LangChain ChatPromptTemplate ───────────────────────────
SYSTEM_PROMPT = """Tu es l'assistant culturel de Puls-Events, une plateforme de découverte
d'événements culturels en Île-de-France et Hauts-de-France.

Ton rôle est d'aider les utilisateurs à trouver des événements culturels adaptés à leurs envies.

Règles :
- Réponds UNIQUEMENT en te basant sur les événements fournis dans le contexte.
- Si aucun événement ne correspond, dis-le clairement.
- Présente les événements de façon claire : titre, lieu, dates, lien si disponible.
- Réponds toujours en français.
- Ne mentionne jamais le contexte ou les scores de similarité.
- Si plusieurs événements correspondent, présente-les en liste numérotée.

Contexte des événements :
{context}"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


# ── Chargement index FAISS ────────────────────────────────────────────────────
def load_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


# ── Embedding requête via LangChain MistralAIEmbeddings ──────────────────────
def embed_query(embeddings_model: MistralAIEmbeddings, texte: str) -> np.ndarray:
    vec = embeddings_model.embed_query(texte)
    return np.array([vec], dtype="float32")


# ── Recherche FAISS ───────────────────────────────────────────────────────────
def search(index, metadata: list, vecteur: np.ndarray, top_k: int = TOP_K):
    distances, indices = index.search(vecteur, k=top_k)
    resultats = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        meta = metadata[idx].copy()
        meta["score"] = float(dist)
        resultats.append(meta)
    return resultats


# ── Conversion résultats en Documents LangChain ───────────────────────────────
def build_langchain_docs(resultats: list) -> list[Document]:
    """Convertit les résultats FAISS en objets Document LangChain."""
    docs = []
    for r in resultats:
        contenu = (
            f"Titre      : {r.get('titre', 'N/A')}\n"
            f"Catégorie  : {r.get('categorie', 'N/A')}\n"
            f"Ville      : {r.get('ville', 'N/A')} ({r.get('region', 'N/A')})\n"
            f"Adresse    : {r.get('adresse', 'N/A')}\n"
            f"Dates      : {r.get('date_debut', 'N/A')} → {r.get('date_fin', 'N/A')}\n"
            f"URL        : {r.get('url', 'N/A')}\n"
            f"Description: {r.get('texte', '')[:400]}"
        )
        docs.append(Document(
            page_content=contenu,
            metadata={
                "titre":    r.get("titre", ""),
                "ville":    r.get("ville", ""),
                "region":   r.get("region", ""),
                "score":    r.get("score", 0),
                "url":      r.get("url", ""),
            }
        ))
    return docs


# ── Construction contexte depuis Documents LangChain ─────────────────────────
def build_context(resultats: list) -> str:
    docs = build_langchain_docs(resultats)
    return "\n\n---\n\n".join(
        [f"Événement {i+1} :\n{doc.page_content}" for i, doc in enumerate(docs)]
    )


# ── Génération via LangChain Chain ────────────────────────────────────────────
def generate_response(llm: ChatMistralAI, question: str, contexte: str) -> str:
    """Génère une réponse via une LangChain chain : prompt | llm | parser."""
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({
        "context":  contexte,
        "question": question,
    })


# ── Pipeline RAG complet ──────────────────────────────────────────────────────
def rag_query(question: str, llm: ChatMistralAI,
              embeddings_model: MistralAIEmbeddings,
              index, metadata: list) -> dict:
    """
    Pipeline RAG complet avec LangChain :
    question → embedding → FAISS → Document → chain LangChain → réponse
    """
    # 1. Embedding via LangChain
    vecteur = embed_query(embeddings_model, question)

    # 2. Recherche FAISS
    resultats = search(index, metadata, vecteur)

    # 3. Conversion en Documents LangChain
    docs = build_langchain_docs(resultats)

    # 4. Contexte
    contexte = "\n\n---\n\n".join(
        [f"Événement {i+1} :\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    # 5. Génération via LangChain chain
    reponse = generate_response(llm, question, contexte)

    return {
        "question": question,
        "reponse":  reponse,
        "sources":  resultats,
        "docs":     docs,
    }


# ── Initialisation ────────────────────────────────────────────────────────────
def init_rag():
    """Initialise le LLM LangChain, les embeddings LangChain et l'index FAISS."""
    llm = ChatMistralAI(
        model=CHAT_MODEL,
        api_key=MISTRAL_API_KEY,
        temperature=0.3,
        max_tokens=1024,
    )
    embeddings_model = MistralAIEmbeddings(
        model=EMBED_MODEL,
        api_key=MISTRAL_API_KEY,
    )
    index, metadata = load_index()
    return llm, embeddings_model, index, metadata