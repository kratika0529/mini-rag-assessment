import streamlit as st
import os
import google.generativeai as genai
from pinecone import Pinecone
import cohere
from dotenv import load_dotenv
import time

# 1. Load Environment Variables
load_dotenv()

# 2. Setup API Keys & Clients
# Google (Gemini)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Pinecone (Vector DB)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "mini-rag"

# Connect to the index (Create it in Pinecone dashboard first: Dim=768, Metric=Cosine)
try:
    index = pc.Index(index_name)
except Exception as e:
    st.error(f"Error connecting to Pinecone. Did you create the index 'mini-rag' with 768 dimensions? \nDetails: {e}")

# Cohere (Reranker)
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# --- APP LAYOUT ---
st.set_page_config(page_title="Mini RAG (Free)", layout="wide")
st.title("📚 Mini RAG: AI Research Assistant")
st.markdown("Powered by **Gemini**, **Pinecone**, and **Cohere**.")

# --- SIDEBAR: SETTINGS & UPLOAD ---
with st.sidebar:
    st.header("1. Upload Knowledge")
    st.info("Paste text or upload a file to add to the knowledge base.")
    
    # Text Input
    raw_text = st.text_area("Paste Text Here:", height=150)
    uploaded_file = st.file_uploader("Or upload .txt file", type=["txt"])
    
    # Process Upload
    if st.button("Add to Knowledge Base"):
        text_to_process = ""
        if raw_text:
            text_to_process += raw_text
        if uploaded_file:
            text_to_process += uploaded_file.read().decode("utf-8")
            
        if text_to_process:
            with st.spinner("Chunking & Embedding..."):
                # A. Simple Chunking Strategy (1000 chars ~ 200-250 tokens)
                chunk_size = 1000
                overlap = 100
                chunks = []
                for i in range(0, len(text_to_process), chunk_size - overlap):
                    chunks.append(text_to_process[i:i + chunk_size])
                
                # B. Create Embeddings (Google Gemini Embedding)
                # Model: models/text-embedding-004 (Output Dim: 768)
                vectors = []
                for i, chunk in enumerate(chunks):
                    response = genai.embed_content(
                        model="models/text-embedding-004",
                        content=chunk,
                        task_type="retrieval_document"
                    )
                    embedding = response['embedding']
                    
                    # Store metadata for citation
                    vectors.append({
                        "id": f"chunk_{int(time.time())}_{i}",
                        "values": embedding,
                        "metadata": {"text": chunk}
                    })
                
                # C. Upsert to Pinecone
                if vectors:
                    index.upsert(vectors=vectors)
                    st.success(f"Successfully added {len(vectors)} chunks to Pinecone!")
        else:
            st.warning("Please provide some text first.")

# --- MAIN AREA: Q&A ---
st.header("2. Ask Questions")
query = st.text_input("What do you want to know?")

if st.button("Get Answer") and query:
    start_time = time.time()
    
    with st.spinner("Searching & Thinking..."):
        # Step 1: Retrieval (Vector Search)
        # Embed query
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
        # Query Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=10, # Get top 10 candidates
            include_metadata=True
        )
        
        # Extract text for Reranking
        retrieved_docs = [match['metadata']['text'] for match in search_results['matches']]
        
        # Step 2: Reranking (Cohere)
        if retrieved_docs:
            rerank_results = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=retrieved_docs,
                top_n=3 # Keep top 3 best chunks
            )
            
            # Prepare context for LLM
            context_text = ""
            top_chunks = [] # To show in UI
            
            for idx, result in enumerate(rerank_results.results):
                # result.index tells us which of the retrieved_docs was picked
                chunk_text = retrieved_docs[result.index]
                context_text += f"Source [{idx+1}]: {chunk_text}\n\n"
                top_chunks.append(chunk_text)
            
            # Step 3: Generation (Gemini LLM)
            # System Prompt
            prompt = f"""
            You are a helpful assistant. Answer the user's question based ONLY on the context provided below.
            If the answer is not in the context, say "I don't know based on the provided text."
            
            CITE YOUR SOURCES. When you use information from a source, put the source number like [1] or [2] at the end of the sentence.
            
            CONTEXT:
            {context_text}
            
            QUESTION:
            {query}
            """
            
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            
            # --- DISPLAY RESULTS ---
            st.markdown("### Answer:")
            st.write(response.text)
            
            # Stats
            end_time = time.time()
            st.caption(f"⏱️ Time taken: {round(end_time - start_time, 2)}s")
            
            # Citations / Sources
            with st.expander("View Source Snippets (Citations)"):
                for i, chunk in enumerate(top_chunks):
                    st.markdown(f"**[{i+1}]** {chunk}")
                    st.divider()
        else:
            st.warning("No relevant information found in the database.")