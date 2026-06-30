"""
04_demo_chatbot.py
Puls-Events - Interactive RAG Chatbot Demo
"""

from rag_chatbot import init_rag, rag_query

BANNER = """
======================================================================
         PULS-EVENTS — RAG Cultural Assistant
     Ile-de-France & Hauts-de-France — Recent Events
======================================================================
  Type your question about cultural events.
  Commands: 'sources' = view sources | 'quit' = exit
"""

def main():
    print(BANNER)
    print("Loading RAG system ...")
    
    try:
        llm, embeddings_model, index, metadatas = init_rag()
        print(f"System ready - {len(metadatas)} events loaded\n")
    except Exception as e:
        print(f"Failed to load RAG system: {e}")
        return

    last_results = []

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye")
            break

        if question.lower() == "sources":
            if not last_results:
                print("No search performed yet.\n")
            else:
                print("\nSources used:")
                for i, r in enumerate(last_results, 1):
                    titre = r.get('titre', r.get('title', 'Sans titre'))
                    ville = r.get('ville', r.get('city', 'Lieu inconnu'))
                    score = r.get('score', 0)
                    print(f"  [{i}] {titre} — {ville} — Score: {score:.4f}")
                print()
            continue

        print("\nSearching...\n")
        try:
            resultat = rag_query(question, llm, embeddings_model, index, metadatas)
            last_results = resultat['sources']
            
            # Afficher la réponse
            reponse = resultat.get('reponse', resultat.get('response', 'Pas de réponse'))
            print(f"Assistant: {reponse}\n")
            print("-" * 60 + "\n")
            
        except KeyError as e:
            print(f"Key error - available keys: {resultat.keys() if 'resultat' in locals() else 'unknown'}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()