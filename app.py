import streamlit as st
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.base import Embeddings
from groq import Groq
from rank_bm25 import BM25Okapi

# ----------------------------
# GROQ CLIENT
# ----------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# ----------------------------
# STREAMLIT CONFIG
# ----------------------------
st.set_page_config(page_title="Agentic RAG Assistant")
st.title("📚 AI Document Assistant")

debug_mode = st.checkbox("Show Debug Mode")

# ----------------------------
# EMBEDDINGS
# ----------------------------
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


# ----------------------------
# LOAD SYSTEM
# ----------------------------
@st.cache_resource
def load_system():

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_function = SentenceTransformerEmbeddings(model)

    loader = PyPDFLoader("data/sample.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local(
            "faiss_index",
            embedding_function,
            allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(chunks, embedding_function)
        vectorstore.save_local("faiss_index")

    # BM25
    chunk_texts = [doc.page_content for doc in chunks]
    tokenized_corpus = [text.split() for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # Reranker
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return vectorstore, bm25, chunks, reranker


vectorstore, bm25, chunks, reranker = load_system()

# ----------------------------
# USER INPUT
# ----------------------------
query = st.text_input("Ask a question about the document:")

if query:

    with st.spinner("Thinking..."):

        # ----------------------------
        # RETRIEVAL
        # ----------------------------
        semantic_docs = vectorstore.similarity_search(query, k=10)

        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)

        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:10]

        keyword_docs = [chunks[i] for i in top_bm25_indices]

        combined_docs = semantic_docs + keyword_docs
        unique_docs = {
            doc.page_content: doc for doc in combined_docs
        }.values()

        docs = list(unique_docs)

        if not docs:
            st.subheader("Answer")
            st.write("The document does not specify this.")
            st.stop()

        # ----------------------------
        # RERANK
        # ----------------------------
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)

        reranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        docs = [doc for doc, score in reranked[:3]]

        # ----------------------------
        # CONTEXT
        # ----------------------------
        context = ""
        for i, doc in enumerate(docs):
            context += f"\nSource {i+1}:\n{doc.page_content}\n"

        # ----------------------------
        # DEBUG
        # ----------------------------
        if debug_mode:
            st.subheader("Retrieved Chunks")
            for i, doc in enumerate(docs):
                st.write(f"Chunk {i+1}:")
                st.write(doc.page_content[:500])
                st.markdown("---")

        # ----------------------------
        # FINAL PROMPT (BALANCED FIX)
        # ----------------------------
        SYSTEM_PROMPT = """
You are answering questions about a novel.

Rules:
1. Use ONLY the provided context
2. Prefer the MAIN story (main character Santiago, not historical/religious references)
3. Answer directly and briefly (1–2 sentences)
4. You MAY use simple obvious understanding from the text
5. DO NOT add outside knowledge
6. If the answer is clearly not present, say:
   "The document does not specify this."
"""

        prompt = f"""
{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = completion.choices[0].message.content.strip()

    # ----------------------------
    # OUTPUT
    # ----------------------------
    st.subheader("Answer")
    st.write(answer)