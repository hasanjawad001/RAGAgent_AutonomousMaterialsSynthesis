##
from pathlib import Path
import re
import fitz 
import tiktoken
from openai import OpenAI
import numpy as np
import faiss
import streamlit as st
import pickle
import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pycountry
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document

##
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def remove_junk_sections(text, section_markers=None):
    if section_markers is None:
        section_markers = [
            "references", 
            "acknowledgment", "acknowledgement", "acknowledgments", "acknowledgements", 
            "author information", "author contribution", "author contributions", 
            "associated content",
        ]
    pattern = re.compile(rf"^\s*[\.\-\u25A0\u2022]*\s*({'|'.join(section_markers)})", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    return text[:match.start()] if match else text

def remove_junk_lines(text, junk_patterns=None):
    if junk_patterns is None:
        junk_patterns = [
            "doi:", "et al.", "https://", "http://", ".org", ".com", "conflict of interest", "bio:", "funding",
            "journal", "citation", "cc-by", "preprint", "arxiv",
            "license", "open access", 
            "submitted to", "peer review", "double-blind", "published", 
            "copyright", "all rights reserved",
            ## "figure", "table",
            "correspondence should be addressed to",
            "authors contributed equally",
            "this manuscript has been authored by",
            "contract no", 
            "supporting information",
            "read online",
            "received:"            
            "revised:",
            "accepted:",
            "cite this:",
            "©",
            "download",
            "public access plan",
            "department of",               
            "university",                  
            "national laboratory",         
            "laboratory",                  
            "government",                              
        ]
    countries = [country.name.lower() for country in pycountry.countries]        
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        lower = line.lower()
        if any(pat in lower for pat in junk_patterns):
            # print('junk: ', line)
            # print()
            continue ## skip if contains junk patterns
        if "*" in line or "†" in line or "∇" in line:
            # print('author name: ', line)            
            # print()            
            continue ## skip if author name 
        ##
        if any(re.search(rf"\b{re.escape(c)}\b", lower) for c in countries):
            # print('author address: ', line)            
            # print()                        
            continue  # skip if author address
        ##
        cleaned.append(line)
    return "\n".join(cleaned)    

def chunk_text(text, max_tokens=None, tokenizer=None):
    if max_tokens is None:
        max_tokens = TOKENS_PER_CHUNK
    if tokenizer is None:
        tokenizer = enc
    words = text.split()
    chunks = []
    current = []
    for word in words:
        current.append(word)
        test_chunk = " ".join(current)
        if len(tokenizer.encode(test_chunk)) > max_tokens:
            current.pop()
            chunks.append(" ".join(current))
            current = [word]
    if current:
        chunks.append(" ".join(current))
    return chunks

def chunk_text2(text, max_tokens=None, tokenizer=None, overlap=None):
    if max_tokens is None:
        max_tokens = TOKENS_PER_CHUNK
    if tokenizer is None:
        tokenizer = enc
    if overlap is None:
        overlap = WORDS_PER_CHUNK_OVERLAP
    words = text.split()
    chunks = []
    current = []
    i=0
    while i<len(words):
        word = words[i]
        current.append(word)
        test_chunk = " ".join(current)
        if len(tokenizer.encode(test_chunk)) > max_tokens:
            current.pop()
            chunks.append(" ".join(current))
            overlap_start = max(len(current) - overlap, 0)
            current = current[overlap_start:] + [word]
        i+=1
    if current:
        chunks.append(" ".join(current))
    return chunks

def estimate_embedding_cost(token_count, model="text-embedding-3-small"):
    price_per_million = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-ada-002": 0.10,
    }.get(model, 0.02) # price per 1 000 000 tokens
    return (token_count / 1000000) * price_per_million
    
## 
## remove 
# input_dir = "inputs"
# output_dir = "outputs"
# pdf_dir_name = os.path.join(input_dir, "content/pdfs/temp")
# gpt_model = "gpt-4o"
# metadata_name = os.path.join(output_dir, "chunk_metadata.pkl")
# embedding_model = "text-embedding-3-small"
# dimension = d_em2dim[embedding_model]
# knowledge_base_name = os.path.join(output_dir, "Test_knowledgebase.index")
##
enc = tiktoken.get_encoding("cl100k_base")
##
d_em2dim = {
    "text-embedding-3-small": 1536,    
    "text-embedding-3-large": 3072
}
TOKENS_PER_CHUNK = 300
WORDS_PER_CHUNK_OVERLAP = int(TOKENS_PER_CHUNK/10) # ~10%
top_k = 50

## s0: init
# st.title("📄 Agent RHEEDiculous")
st.markdown("## 📄 RAG Agent - Autonomous Synthesis")
st.divider()

## s1: api key and verification
if "api_verified" not in st.session_state:
    st.session_state.api_verified = False
if "client" not in st.session_state:
    st.session_state.client = None
if not st.session_state.api_verified:
    with st.form("api_form"):
        api_key = st.text_input("🔑 Enter your OpenAI API key", type="password")
        verify_btn = st.form_submit_button("Verify")
        if verify_btn:
            try:
                client = OpenAI(api_key=api_key)
                _ = client.models.list()  # Check API validity
                st.session_state.api_verified = True
                st.session_state.client = client
                st.session_state.api_key = api_key
                st.success("✅ API key verified!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Invalid API key: {e}")
    st.stop()
client = st.session_state.client

## s2: knowledge base setup
st.header("📂 Knowledge Base Setup")
option = st.radio(
    "Choose how you want to set up the knowledge base:",
    ("📥 Build a new knowledge base", "📤 Load existing knowledge base"),
    index=0
)
## build
if option == "📥 Build a new knowledge base":
    st.session_state.embedding_model = st.selectbox(
        "🔍 Select your embedding model:",
        options=["text-embedding-3-small", "text-embedding-3-large"],
        index=0,
        help="This model is used for generating embeddings → impacts context matching"
    )
    st.session_state.dimension = d_em2dim[st.session_state.embedding_model] 
    st.session_state.pdf_folder = st.text_input("📁 Input Folder path for PDFs:", value='inputs/content/pdfs/temp', help="Path to get the PDFs as input to build the knowledge base")    
    st.session_state.knowledge_base_name = st.text_input("🧠 Output FAISS index file path (.index)", value='outputs/Test_knowledgebase.index', help="Path to save the FAISS index")
    st.session_state.metadata_name = st.text_input("📝 Output metadata file path (.pkl)", value='outputs/chunk_metadata.pkl', help="Path to save metadata for chunks")
    if st.button("📚 Build Knowledge Base"):
        ##
        ## get chunks
        if not os.path.isdir(st.session_state.pdf_folder):
            st.error("The provided folder path does not exist.")
            st.stop()
        pdf_files = list(Path(st.session_state.pdf_folder).glob("*.pdf"))
        if not pdf_files:
            st.error("No PDF files found in the selected folder.")
            st.stop()
        total_chunks = 0
        total_tokens = 0
        total_cost = 0.0
        all_chunks = {}
        st.write("🔍 Processing PDFs...")
        progress_bar = st.progress(0)
        for ipp, pdf_path in enumerate(pdf_files):
            try:
                text = extract_text_from_pdf(pdf_path)
                text = remove_junk_sections(text)
                text = remove_junk_lines(text)
                chunks = chunk_text2(text, max_tokens=TOKENS_PER_CHUNK, tokenizer=enc, overlap=WORDS_PER_CHUNK_OVERLAP)
                token_count = sum(len(enc.encode(chunk)) for chunk in chunks)
                cost_emb = estimate_embedding_cost(token_count, model=st.session_state.embedding_model)
                cost = cost_emb
                total_chunks += len(chunks)
                total_tokens += token_count
                total_cost += cost
                all_chunks[pdf_path.name] = chunks
                # st.write(f"✅ {pdf_path.name}: {len(chunks)} chunks, {token_count:,} tokens.")
                # st.write(f"✅ {pdf_path.name}: {len(chunks)} chunks, {token_count:,} tokens, ~${cost:.6f}")
            except Exception as e:
                st.warning(f"⚠️ Failed on {pdf_path.name}: {e}")
            progress_percent = int((ipp + 1) / len(pdf_files) * 100)
            progress_bar.progress(progress_percent)
        st.write(f"📦 Total chunks: {total_chunks:,}, Total tokens: {total_tokens:,}.")        
        # st.write(f"📦 Total tokens: {total_tokens:,} | Est. cost: ${total_cost:.6f}")
        # st.write(f'ℹ️ Actual cost may vary depending on caching, usage, or pricing changes.')
        ##
        ## get embeddings
        st.write("📌 Embedding and indexing...")
        all_embeddings = []
        all_metadata = []
        chunk_count = 0
        progress_bar = st.progress(0)
        for filename, chunks in all_chunks.items():
            for i, chunk in enumerate(chunks):
                try:
                    response = client.embeddings.create(input=chunk, model=st.session_state.embedding_model)
                    embedding = response.data[0].embedding
                    all_embeddings.append(embedding)
                    all_metadata.append({
                        "source": filename,
                        "chunk_id": i,
                        "text": chunk,
                        "embedding_model": st.session_state.embedding_model
                    })
                    # st.write(f"Embedded: {filename}, chunk_id: {i}")
                except Exception as e:
                    st.warning(f"⚠️ Embedding failed: {filename}, chunk_id: {i}, error: {e}")
                chunk_count += 1
                progress = int((chunk_count / total_chunks) * 100)
                progress_bar.progress(progress)
        embedding_matrix = np.array(all_embeddings, dtype="float32")
        st.session_state.dimension = d_em2dim[st.session_state.embedding_model] 
        index = faiss.IndexFlatL2(st.session_state.dimension)
        index.add(embedding_matrix)
        os.makedirs(os.path.dirname(st.session_state.knowledge_base_name), exist_ok=True)
        os.makedirs(os.path.dirname(st.session_state.metadata_name), exist_ok=True)        
        faiss.write_index(index, st.session_state.knowledge_base_name)
        with open(st.session_state.metadata_name, "wb") as f:
            pickle.dump(all_metadata, f)
        st.success("✅ Knowledge base built and saved!")
        st.session_state.index = index
        st.session_state.metadata = all_metadata
        ## docstore
        metadata = all_metadata
        # build a docstore and ID map
        ids = [str(i) for i in range(len(metadata))]
        docs_dict = {
            ids[i]: Document(
                page_content=meta["text"],
                metadata={"source":meta["source"], "chunk_id":meta["chunk_id"]}
            )
            for i, meta in enumerate(metadata)
        }
        docstore = InMemoryDocstore(docs_dict)
        index_to_docstore_id = {i: ids[i] for i in range(len(ids))}
        # instantiate the FAISS wrapper (embedding_function is a no-op
        # because we only call search-by-vector methods)
        db = FAISS(
            embedding_function=lambda _: [],  
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
        st.session_state.db = db
        ##
## load
elif option == "📤 Load existing knowledge base":
    index_path = st.text_input("🧠 Index file path (.index)", value='outputs/Test_knowledgebase.index')
    meta_path = st.text_input("📝 Metadata file path (.pkl)", value='outputs/chunk_metadata.pkl')
    if st.button("📂 Load Knowledge Base"):
        if not os.path.isfile(index_path):
            st.error("Index file not found.")
            st.stop()
        if not os.path.isfile(meta_path):
            st.error("Metadata file not found.")
            st.stop()
        try:
            index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            ## docstore
            # build a docstore and ID map
            ids = [str(i) for i in range(len(metadata))]
            docs_dict = {
                ids[i]: Document(
                    page_content=meta["text"],
                    metadata={"source":meta["source"], "chunk_id":meta["chunk_id"]}
                )
                for i, meta in enumerate(metadata)
            }
            docstore = InMemoryDocstore(docs_dict)
            index_to_docstore_id = {i: ids[i] for i in range(len(ids))}
            # instantiate the FAISS wrapper (embedding_function is a no-op
            # because we only call search-by-vector methods)
            db = FAISS(
                embedding_function=lambda _: [],  
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
            )
            st.session_state.db = db
            ##
            st.session_state.embedding_model = metadata[0].get("embedding_model", "text-embedding-3-small")
            st.session_state.dimension = d_em2dim[st.session_state.embedding_model] 
            st.session_state.index = index
            st.session_state.metadata = metadata
            st.success(f"✅ Knowledge base loaded successfully! Embedding model: `{st.session_state.embedding_model}`")            
        except Exception as e:
            st.error(f"❌ Failed to load knowledge base: {e}")
            st.stop()
st.divider()

## s3: qa
st.header("❓ Ask a Question")
st.session_state.gpt_model = st.selectbox(
    "🤖 Select GPT model:",
    options=[
        "gpt-4o-2024-08-06",
        "gpt-4.1-2025-04-14",
        "o4-mini-2025-04-16",
        "o4-mini-deep-research-2025-06-26"
    ],
    index=0,
    help="This model is ussed for generating responses"
)
with st.expander("🎛️ Advanced Controls: Diversity & Creativity"):
    diversity = st.slider(
        "🧭 Diversity Radar (0 = Homogeneous, 1 = Diverse)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
        help="Controls diversity in context. Higher values prioritize more varied sources; lower values focus less on source diversity."
    )
    if st.session_state.gpt_model in ["gpt-4o-2024-08-06", "gpt-4.1-2025-04-14"]:
        temperature = st.slider(
            "🔥 Creativity Dial (0 = Boring, 1 = Creative)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help="Lower values make the model more robotic and safe; higher values make it more imaginative and expressive."
        )
    else:
        temperature = 0.3
    
if "index" in st.session_state:
    query = st.text_area("Ask your question here:", height=300)    
    if st.button("💬 Answer") and query:
        query_embedding = client.embeddings.create(
            input=query,
            model=st.session_state.embedding_model
        ).data[0].embedding
        query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
        ## plain vector base index search
        # D, I = st.session_state.index.search(query_embedding, k=top_k)
        # context_meta_chunks = []
        # for idx in I[0]:
        #     meta = st.session_state.metadata[idx]
        #     context_meta_chunks.append(f"[{meta['source']} | chunk {meta['chunk_id']}]: {meta['text']}")
        ## MMR
        flat_emb = query_embedding[0].astype(float).tolist()
        # run FAISS+MMR in one call
        results = st.session_state.db.max_marginal_relevance_search_by_vector(
            embedding=flat_emb,
            k=top_k,         # e.g. 50
            fetch_k=1000,    # or whatever raw window you want, reranking will be applied on that
            lambda_mult=1.0 - diversity  # your chosen trade‐off (0 more diverse, 1 less diverse)
        )
        # build context from the returned Documents
        context_meta_chunks = [
            f"[{doc.metadata['source']} | chunk {doc.metadata['chunk_id']}]: {doc.page_content}"
            for doc in results
        ]
        ##
        context_meta = "\n\n".join(context_meta_chunks)
        prompt = f"""
User query: {query}

--- BEGIN CONTEXT ---
{context_meta}
--- END CONTEXT ---

Answer:
"""
        system_instructions = "You are an expert scientific research assistant. Use the context provided from research papers to answer the user query as accurately as possible. You provide detailed responses using as much of the provided context as possible, giving background information when needed to support your response. If the answer is not clearly found in the context, respond with: 'The context does not provide enough information to answer this question.' If the provided context is not enough to answer the user, use web search to find revelent information.At the end of your answer, also mention the source file name(s) where the answer came from, as listed in the context before each chunk (e.g., [DL-rheed-harris-SI.pdf | chunk 1]) or with citations of paper you found online."
        if st.session_state.gpt_model=="o4-mini-deep-research-2025-06-26":
            response = client.responses.create(
                model=st.session_state.gpt_model,         # e.g. "o4-mini-deep-research-2025-06-26"
                instructions=system_instructions,
                tools=[{ "type": "web_search_preview" }],
                input=prompt,
            )
            answer_text = response.output_text
        elif st.session_state.gpt_model=="o4-mini-2025-04-16":
            response = client.chat.completions.create(
              model=st.session_state.gpt_model,
              messages=[{"role":"system","content": system_instructions}, {"role": "user", "content": prompt}],
            )
            answer_text = response.choices[0].message.content
        else:
            response = client.chat.completions.create(
                model=st.session_state.gpt_model,
                messages=[{"role":"system","content": system_instructions}, {"role": "user", "content": prompt}],
                temperature=temperature,
            )
            answer_text = response.choices[0].message.content
        st.markdown("### 💡 Answer")
        st.write(answer_text.strip())
        with st.expander("📚 Retrieved Context for this Query"):
            context_meta_html = context_meta.replace("\n", "<br>")
            st.markdown(f"<div style='overflow-wrap: break-word; width: 600px'>{context_meta_html}</div>", unsafe_allow_html=True)
else:
    st.info("⚠️ Please build or load a knowledge base before asking a question.")
