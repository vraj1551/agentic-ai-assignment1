import os
import numpy as np
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Your Groq API key from environment
API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)

# Knowledge base (you can expand this)
docs = [
    "Argentina won the FIFA World Cup in 2022.",
    "The capital of France is Paris.",
    "Groq provides ultra-fast inference for LLMs.",
    "Max Verstappen won the Formula 1 Drivers Championship in 2024.",
    "Red Bull Racing won the Formula 1 Constructors Championship in 2024.",
    "Cristiano Ronaldo has scored more than 850 career goals across club and country.",
    "The Great Wall of China is over 21,000 kilometers long.",
    "Python is a popular programming language created by Guido van Rossum in 1991.",
    "Mount Everest is the tallest mountain in the world, standing at 8,849 meters.",
    "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
    "The Amazon Rainforest produces 20% of the world's oxygen supply.",
    "India became independent from British rule on August 15, 1947.",
    "The speed of light is approximately 299,792 kilometers per second.",
    "The human brain has around 86 billion neurons.",
    "The Taj Mahal, located in Agra, India, was built by Mughal Emperor Shah Jahan.",
]
doc_ids = [f"doc{i+1}" for i in range(len(docs))]

# Precompute document embeddings
doc_embeddings = embedder.encode(docs)

def retrieve_top_k(query: str, k=3):
    query_emb = embedder.encode([query])
    scores = cosine_similarity(query_emb, doc_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:k]
    top_docs = [docs[i] for i in top_indices]
    top_ids = [doc_ids[i] for i in top_indices]
    return top_docs, top_ids

def rag_with_groq(query: str) -> str:
    top_docs, top_ids = retrieve_top_k(query)
    context = "\n".join(top_docs)

    system_prompt = f"""
You are a helpful assistant. Use the context below to answer the user's query.

Context:
{context}

Sources: {', '.join(top_ids)}
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
    )

    return response.choices[0].message.content

# CLI test
if __name__ == "__main__":
    query = input("Enter your query: ")
    answer = rag_with_groq(query)
    print("\nAnswer:\n", answer)
