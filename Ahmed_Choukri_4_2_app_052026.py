"""
app.py
Puls-Events - Interface web Streamlit pour chatbot RAG
"""

import streamlit as st
import json
import pickle
import numpy as np
import faiss
import os
from datetime import datetime
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

# Configuration de la page
st.set_page_config(
    page_title="Puls-Events - Assistant Culturel",
    page_icon="",
    layout="wide"
)

# Titre
st.title("Puls-Events")
st.markdown("### Assistant Culturel - Île-de-France & Hauts-de-France")
st.markdown("---")

# Cache pour éviter de recharger à chaque interaction
@st.cache_resource
def load_system():
    """Charge l'index FAISS et les données"""
    with st.spinner("Chargement de la base d'événements..."):
        index = faiss.read_index("faiss_index.bin")
        with open("faiss_metadata.pkl", "rb") as f:
            metadatas = pickle.load(f)
        
        with open('data_processed.json', 'r', encoding='utf-8') as f:
            events = {e['id']: e for e in json.load(f)}
        
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            st.error("MISTRAL_API_KEY non trouvée dans .env")
            st.stop()
        
        client = Mistral(api_key=api_key)
        
        return client, index, metadatas, events

# Fonction pour générer une réponse avec Mistral
def generate_response(query, events_list):
    """Génère une réponse en langage naturel"""
    if not events_list:
        return "Désolé, je n'ai pas trouvé d'événements correspondant à votre recherche."
    
    # Construction du contexte
    context = ""
    for i, evt in enumerate(events_list[:5], 1):
        context += f"""
[{i}] {evt.get('title', 'Sans titre')}
    Lieu : {evt.get('city', 'Non spécifié')}
    Dates : {evt.get('date_start', 'Date non spécifiée')}
    Description : {evt.get('description', '')[:200]}
"""
    
    prompt = f"""Vous êtes un assistant culturel spécialisé dans les événements en Île-de-France et Hauts-de-France.

Voici les événements pertinents pour répondre à la question :

{context}

Question : {query}

Répondez de manière naturelle et utile en français. Citez les événements avec leurs dates et lieux."""

    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        # Réponse de secours simple
        fallback = f"J'ai trouvé {len(events_list)} événements :\n\n"
        for i, evt in enumerate(events_list[:3], 1):
            fallback += f"{i}. **{evt.get('title', 'Sans titre')}**\n"
            fallback += f"   Lieu : {evt.get('city', 'Lieu inconnu')}\n"
            if evt.get('date_start'):
                fallback += f"   Date : {evt.get('date_start')}\n"
            fallback += "\n"
        return fallback

# Fonction de recherche vectorielle
def search_events(query, index, client, metadatas, events_data, k=5):
    """Recherche des événements similaires"""
    try:
        response = client.embeddings.create(
            model="mistral-embed",
            inputs=[query]
        )
        query_embedding = np.array([response.data[0].embedding]).astype('float32')
        distances, indices = index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(metadatas):
                meta = metadatas[idx]
                event = events_data.get(meta.get('event_id', ''), {})
                results.append({
                    'title': meta.get('title', 'Sans titre'),
                    'city': meta.get('city', ''),
                    'region': meta.get('region', ''),
                    'category': meta.get('category', ''),
                    'date_start': event.get('date_start_fmt', ''),
                    'date_end': event.get('date_end_fmt', ''),
                    'description': event.get('description', '')[:300],
                    'url': event.get('url', ''),
                    'score': float(1.0 / (1.0 + distances[0][i]))
                })
        return results
    except Exception as e:
        st.error(f"Erreur de recherche : {e}")
        return []

# Chargement du système
try:
    from rag_chatbot import init_rag, rag_query
    llm, embeddings_model, index, metadatas = init_rag()
    st.success(f"OK {len(metadatas)} événements disponibles")
    
    # Chargement des données d'événements pour l'affichage
    with open('data_processed.json', 'r', encoding='utf-8') as f:
        events = {e['id']: e for e in json.load(f)}
    
    # Initialisation du client Mistral
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("MISTRAL_API_KEY non trouvée dans .env")
        st.stop()
    client = Mistral(api_key=api_key)
    
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# Barre latérale
with st.sidebar:
    st.header("Filtres")
    
    # Statistiques
    st.markdown("### Statistiques")
    st.metric("Événements", len(metadatas))
    
    # Exemples
    st.markdown("---")
    st.markdown("### Exemples")
    examples = [
        "Concerts gratuits à Paris",
        "Expositions pour enfants",
        "Événements en plein air",
        "Atelier de musique classique",
        "Que faire à Lille en juin ?"
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state.question = ex
            st.rerun()

# Zone principale
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Posez votre question")
    question = st.text_input("", placeholder="Exemple : Où voir un concert de jazz à Paris ?", key="question_input")
    
    col_q1, col_q2 = st.columns([1, 5])
    with col_q1:
        search_clicked = st.button("Rechercher", type="primary", use_container_width=True)

with col2:
    st.markdown("### Suggestions")
    st.info("Concerts\nExpositions\nSpectacles\nAteliers\nFestivals")

# Recherche et affichage des résultats
if search_clicked and question:
    with st.spinner("Recherche d'événements..."):
        results = search_events(question, index, client, metadatas, events, k=5)
    
    if results:
        st.markdown("---")
        st.markdown(f"### Résultats pour : *{question}*")
        
        # Génération de la réponse avec le LLM
        with st.spinner("Génération de la réponse..."):
            answer = generate_response(question, results)
        st.markdown("#### Réponse")
        st.success(answer)
        
        st.markdown("#### Détails des événements")
        
        # Affichage de chaque événement
        for i, evt in enumerate(results, 1):
            with st.expander(f"{i}. {evt['title']} — {evt['city']} (Pertinence : {evt['score']:.2f})"):
                st.markdown(f"**Lieu :** {evt['city']} ({evt['region']})")
                if evt['date_start']:
                    st.markdown(f"**Date :** {evt['date_start']}")
                if evt['category']:
                    st.markdown(f"**Catégorie :** {evt['category']}")
                if evt['description']:
                    st.markdown(f"**Description :** {evt['description']}...")
                if evt['url']:
                    st.markdown(f"**Lien :** [Voir sur OpenAgenda]({evt['url']})")
    else:
        st.warning("Aucun événement trouvé. Essayez une autre question.")

elif search_clicked and not question:
    st.warning("Veuillez entrer une question.")

# Pied de page
st.markdown("---")
st.markdown("*Puls-Events - Système RAG avec Mistral AI & FAISS*")