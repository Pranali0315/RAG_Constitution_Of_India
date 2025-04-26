from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from llama_cpp import Llama
import numpy as np

# 1. Embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Qdrant setup
client = QdrantClient(host="localhost", port=6333)
collection_name = "rag_chunks"

# 3. LLM setup
llm = Llama(
    model_path=r"C:\Users\Rushi\Desktop\RAG proj\models\gemma-2-2b-q5_k_m.gguf",
    n_ctx=4096,
    n_threads=4,
    use_mlock=True
)

# ✨ Utilities
def remove_repetitions(text):
    seen = set()
    result = []
    for sentence in text.split(". "):
        clean = sentence.strip()
        if clean and clean not in seen:
            result.append(clean)
            seen.add(clean)
    return ". ".join(result)

def clean_response(text):
    text = text.strip()
    if "____" in text or len(set(text)) <= 2:
        return "⚠️ The model was unable to generate a valid response."

    lines = text.split("\n")
    seen = set()
    filtered = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            filtered.append(line)
            seen.add(line)
    return remove_repetitions("\n".join(filtered)).strip()


# ✅ MAIN RAG FUNCTION
def run_rag(query, top_k=3, min_score_threshold=0.3):
    # Step 1: Embed the query
    query_vector = embed_model.encode(query).tolist()

    # Step 2: Retrieve chunks from Qdrant
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    if not search_result or search_result[0].score < min_score_threshold:
        return "🛑 The answer is not available in the provided context."

    # Step 3: Build context
    context = "\n\n".join([hit.payload['text'] for hit in search_result])
    print(context)
    # Step 4: Prompt
    prompt = f"""
You are an expert assistant specialized in the Constitution of India.

Use the provided context to answer the user's question **factually and completely**.
Your answer should follow following rules strictly:
- Strictly based on the context
- Clear, helpful, and informative
- Concise for simple questions, detailed if needed
- Do NOT repeat or add extra info
- If context is irrelevant, respond: "The answer is not available in the provided context."

---

📜 Context:
{context}

❓ Question:
{query}

🧾 Answer:
"""

    # Step 5: Generate answer
    output = llm(
        prompt,
        max_tokens=350,
        temperature=0.3,
        top_k=30,
        top_p=0.8,
        stop=["###", "Question:", "Context:", "</s>"]
    )

    response_text = output["choices"][0]["text"]
    return clean_response(response_text)


# 🧠 CLI Test Loop
if __name__ == "__main__":
    print("🧠 Constitution RAG Assistant")
    print("Type 'exit' to end the conversation.\n")
    
    while True:
        user_query = input("🧾 Ask your question: ").strip()
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("👋 Exiting... See you next time!")
            break

        answer = run_rag(user_query)
        print("\n🔍 Answer:\n", answer)
        print("-" * 60)
        