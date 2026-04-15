# 🏥 MedQuery AI — Complete End-to-End Project Analysis

## Table of Contents
1. [Project Overview](#overview)
2. [Problem Statement](#problem)
3. [Technology Stack](#tech-stack)
4. [Core Concepts Explained](#concepts)
5. [System Architecture](#architecture)
6. [Data Flow — Step by Step](#data-flow)
7. [Pipeline Nodes Deep Dive](#nodes)
8. [Evaluation Metrics](#metrics)
9. [Streamlit Frontend](#frontend)
10. [Project Files Breakdown](#files)
11. [Key Advantages vs Naive RAG](#advantages)

---

## 1. 📌 Project Overview {#overview}

**MedQuery AI** is a **Corrective Retrieval-Augmented Generation (CRAG)** system built for Clinical Question Answering. It is designed to assist doctors and clinicians in retrieving accurate, evidence-based medical information from a large knowledge base while actively detecting and correcting irrelevant or hallucinated responses.

> It is a **mini-project** that demonstrates advanced AI techniques: Agentic LangGraph pipelines, vector databases, embedding-based retrieval, LLM-as-judge grading, and real-time observability.

---

## 2. 🎯 Problem Statement {#problem}

### Why not just use a regular chatbot?
Standard LLMs like GPT or Llama hallucinate — they confidently generate incorrect medical facts (wrong dosages, wrong thresholds, wrong drugs). In healthcare, this can be **life-threatening**.

### Why not just use basic RAG?
Basic (Naive) RAG simply retrieves documents and passes them to the LLM — it has **no quality check**:
- Retrieves irrelevant documents → still generates an answer
- No verification that the answer matches the retrieved context
- No fallback when local knowledge base fails

### The CRAG Solution
CRAG adds **self-correction loops** — it grades retrieved documents, rewrites queries, falls back to web search, and verifies the final answer before returning it. This dramatically reduces hallucinations.

---

## 3. 🛠️ Technology Stack {#tech-stack}

| Category | Tool | Purpose |
|---|---|---|
| **LLM** | Groq API + Llama 3.1 8B Instant | Fast, free-tier language model for grading, generation, verification |
| **Orchestration** | LangGraph | Stateful, multi-node agentic pipeline with conditional routing |
| **Prompting** | LangChain Core | ChatPromptTemplate, chains, output parsers |
| **Embeddings** | `all-MiniLM-L6-v2` (Sentence Transformers) | Converts text into 384-dim vectors for similarity search |
| **Vector DB** | ChromaDB | Persists and retrieves embedded document chunks locally |
| **Web Search** | Tavily Search API | Fallback web search when local knowledge fails |
| **Document Loading** | PyPDFDirectoryLoader (LangChain) | Loads medical PDFs from the `data/` directory |
| **Text Splitting** | RecursiveCharacterTextSplitter | Splits documents into 500-char chunks with 50-char overlap |
| **Frontend** | Streamlit | Interactive web UI for queries, metrics, and history |
| **Observability** | LangSmith | Full tracing of every LLM call and pipeline step |
| **Environment** | Python-dotenv | Loads API keys from `.env` file |
| **Metrics** | scikit-learn, numpy | Cosine similarity for answer relevance scoring |

---

## 4. 🧠 Core Concepts Explained {#concepts}

### 4.1 RAG (Retrieval-Augmented Generation)
Instead of relying only on what the LLM was trained on, RAG **retrieves relevant external knowledge** at runtime and provides it as context.

```
User Query → Retrieve Docs → Feed to LLM → Generate Answer
```

### 4.2 Embeddings
Text is converted into **numerical vectors** using a transformer model. Similar texts produce similar vectors. This allows **semantic similarity search** — finding related documents even if they use different words.

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Output: 384-dimensional float vector per text chunk
- Example: "heart attack" and "myocardial infarction" will have close vectors

### 4.3 ChromaDB (Vector Store)
A **local vector database** that stores all document embeddings. At query time, it performs **k-Nearest Neighbour (k-NN)** search to find the top-k most similar document chunks.

```
Medical Book (637 pages) → Chunked (5,961 chunks) → Embedded → Stored in ChromaDB
Query → Embedded → Distance search → Top 5 chunks returned
```

### 4.4 LangGraph (Stateful Agentic Pipeline)
LangGraph treats the CRAG pipeline as a **State Machine** — a directed graph where:
- **Nodes** = processing steps (functions)
- **Edges** = transitions between steps
- **State** = a Python TypedDict that carries data between nodes
- **Conditional Edges** = routing logic (if-else decisions between nodes)

This is more powerful than a simple chain because it allows **loops, retries, and branching**.

### 4.5 LLM-as-Judge (Document Grading)
Instead of a heuristic, the system asks the **LLM itself** to judge whether each retrieved document is relevant to the question. The LLM returns structured JSON:
```json
{"relevance": "RELEVANT", "reason": "Document specifically addresses metformin dosage"}
```

### 4.6 Hallucination Detection
After generating an answer, the system asks the LLM to **verify** whether every claim in the answer is actually supported by the retrieved context:
```json
{"verdict": "grounded", "confidence": 0.95, "ungrounded_claims": []}
```
If claims are not grounded, the system re-routes to web search.

### 4.7 CRAG (Corrective RAG)
CRAG extends RAG with three correction mechanisms:

| Scenario | Action |
|---|---|
| All docs relevant | Directly generate |
| Mixed relevance | **Transform query** and re-retrieve |
| All docs irrelevant | **Web search fallback** (Tavily) |
| Hallucination detected | Re-route to web search |

### 4.8 LangSmith (Observability)
Every LLM call, node execution, and chain invocation is **automatically traced** to LangSmith. You can see the full tree of what happened, how long each step took, what was input/output, and token costs.

---

## 5. 🏗️ System Architecture {#architecture}

```
┌───────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                          │
│  PDF Files (data/) → PyPDFDirectoryLoader → 637 pages             │
│  → RecursiveCharacterTextSplitter (500 chars, 50 overlap)         │
│  → 5,961 chunks → all-MiniLM-L6-v2 → ChromaDB (persisted)        │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼ At Query Time
┌───────────────────────────────────────────────────────────────────┐
│                      CRAG LANGGRAPH PIPELINE                      │
│                                                                   │
│   [CRAGState TypedDict] — shared memory across all nodes          │
│                                                                   │
│   ① RETRIEVE    → k-NN search in ChromaDB (k=5)                  │
│   ② GRADE       → LLM grades each doc: RELEVANT/IRRELEVANT/AMB   │
│   ③ ROUTE       → Conditional edge: generate/transform/websearch  │
│   ④ TRANSFORM   → LLM rewrites query with medical terminology     │
│      (loop back to ①)                                             │
│   ⑤ WEB SEARCH → Tavily API returns 3 medical web results        │
│   ⑥ GENERATE   → Groq Llama 3.1 generates grounded answer        │
│   ⑦ HALLCHECK  → LLM verifies every claim is grounded            │
│      (if not_grounded → loop back to ⑤)                          │
│                                                                   │
│   OUTPUT: Answer + Sources + Pipeline Path + Metrics              │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                      STREAMLIT FRONTEND                           │
│  Tab 1: Query Interface + Pipeline Trace + Answer + Grading       │
│  Tab 2: Live Metrics Dashboard (Recall, Faithfulness, etc.)       │
│  Tab 3: Session History (all queries + metrics)                   │
│  Tab 4: Architecture Diagram                                      │
│  Sidebar: API Keys + PDF Upload + k-docs slider + History         │
└───────────────────────────────────────────────────────────────────┘
```

---

## 6. 🔄 Data Flow — Step by Step {#data-flow}

### Phase A: Data Ingestion (Run Once — `ingest.py`)

```
Step 1: Run python ingest.py
Step 2: PyPDFDirectoryLoader scans data/ folder → loads all PDFs
Step 3: Each page gets metadata (title, doc_id, page number, category)
Step 4: RecursiveCharacterTextSplitter chunks pages
        → chunk_size=500, chunk_overlap=50
        → 5,961 total chunks
Step 5: HuggingFaceEmbeddings converts each chunk → 384-dim vector
        Model: sentence-transformers/all-MiniLM-L6-v2
Step 6: Chroma.from_documents() stores all vectors in chroma_medical_db/
        Collection name: "medical_documents"
```

### Phase B: Query Processing (CRAG Pipeline)

```
Step 1: User enters: "What is the maximum daily dose of metformin?"

Step 2: RETRIEVE
        query → embed → cosine similarity vs all 5961 vectors
        → top 5 most similar chunks returned

Step 3: GRADE DOCUMENTS
        For each of 5 docs, Groq LLM is called:
        "Is this document relevant to the question?"
        → Returns JSON: {relevance: RELEVANT/IRRELEVANT/AMBIGUOUS}

Step 4: ROUTE (Conditional Edge)
        Count relevant vs irrelevant:
        - All 5 RELEVANT           → jump to GENERATE
        - Mix of relevant/ambiguous → TRANSFORM QUERY
        - All 5 IRRELEVANT         → WEB SEARCH
        - retry_count >= 2         → GENERATE anyway (circuit breaker)

Step 5a: TRANSFORM QUERY (if mixed)
        Groq rewrites the query:
        "What is the maximum metformin dose" →
        "metformin maximum recommended daily dosage adult clinical guidelines contraindications"
        → Re-retrieve → Re-grade → Route again

Step 5b: WEB SEARCH (if all irrelevant)
        Tavily API called with: "medical {question}"
        → Returns 3 web results as Document objects
        → Goes directly to GENERATE

Step 6: GENERATE
        Groq Llama 3.1 8B receives:
        - The original question
        - All relevant/ambiguous context chunks joined
        System Prompt: "Answer using ONLY provided context. Include dosages/thresholds."
        → Generates full clinical answer with SOURCES at the end

Step 7: HALLUCINATION CHECK
        Groq called again with answer + context:
        "Is every claim in this answer supported by the context?"
        → Returns: {verdict: "grounded", confidence: 0.95, ungrounded_claims: []}
        - grounded      → Return answer to user
        - not_grounded  → Loop back to WEB SEARCH for better context

Step 8: Return result dict with:
        - answer text
        - graded documents (with relevance labels)
        - pipeline_trace (which nodes were visited)
        - sources used
        - hallucination verdict + confidence
        - metrics (Recall@5, Faithfulness, Relevance, Hallucination Rate)
```

---

## 7. 🔵 Pipeline Nodes Deep Dive {#nodes}

### Node ①: `retrieve_node`
```python
docs = retriever.invoke(query)
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
```
- Uses ChromaDB's cosine similarity to find top-5 chunks
- If `transformed_query` exists in state, uses that instead of original question

### Node ②: `grade_documents_node`
```python
# For each doc, calls Groq with structured JSON prompt
GRADE_PROMPT = "You are a medical relevance grader. Respond ONLY with JSON:
{\"relevance\": \"RELEVANT\"|\"IRRELEVANT\"|\"AMBIGUOUS\", \"reason\": \"...\"}"
```
- Calls LLM once per retrieved document (5 calls for k=5)
- Falls back to `"AMBIGUOUS"` if JSON parsing fails

### Node ③: Routing Function `route_documents(state)`
```python
if retry_count >= 2:      return "generate"   # circuit breaker
if all_relevant:          return "generate"
if all_irrelevant:        return "web_search"
else:                     return "transform_query"
```
- This is a **conditional edge function**, not a node
- Returns a string that LangGraph uses to pick the next node

### Node ④: `transform_query_node`
```python
TRANSFORM_PROMPT = "Rewrite this medical query with precise clinical terminology. Return ONLY the rewritten query."
# Increments retry_count to prevent infinite loops
```
- After transformation, LangGraph automatically loops back to `retrieve_node`

### Node ⑤: `web_search_node`
```python
results = web_search_tool.invoke({"query": f"medical {question}"})
# Returns 3 Tavily web results converted to Document objects
```
- Only triggered when all local docs are irrelevant
- Also triggered when hallucination is detected and retry < 2

### Node ⑥: `generate_node`
```python
GEN_PROMPT = """You are an expert medical AI assistant for clinicians.
Answer using ONLY the provided context. Include specific values (dosages, thresholds).
End with: SOURCES: [document titles used]"""
```
- Context window includes top 5 relevant/ambiguous docs
- Sources are extracted from document metadata and included in answer

### Node ⑦: `hallucination_check_node`
```python
HALL_PROMPT = """Verify if this medical answer is grounded in the context.
Respond ONLY with JSON: {"verdict": "grounded"|"not_grounded",
"confidence": 0.0-1.0, "ungrounded_claims": [...]}"""
```
- Acts as a safety gate before returning the answer
- If `not_grounded` and `retry_count < 2` → re-routes to web search

---

## 8. 📊 Evaluation Metrics {#metrics}

The system computes 4 real-time metrics per query:

| Metric | Formula | What it measures |
|---|---|---|
| **Recall@5** | `hits / total docs retrieved` | Were the right docs in the top 5? |
| **Faithfulness** | `answer_words ∩ context_words / answer_words × 2.5` | Does the answer use words from context? |
| **Answer Relevance** | `cosine_similarity(embed(question), embed(answer))` | Is the answer semantically related to question? |
| **Hallucination Rate** | `0 if grounded, 1 if not` | Did the hallucination check pass? |

### The Notebook also runs Naive RAG vs CRAG comparison:
- Same 20 clinical QA questions for both
- Naive RAG: retrieve → generate (no grading, no correction)
- CRAG: full 7-node pipeline
- CRAG consistently outperforms on Faithfulness and Hallucination Rate

---

## 9. 💻 Streamlit Frontend {#frontend}

### Tab 1: Query Interface
- Text area for clinical question input
- 8 pre-built example queries
- **Run CRAG Analysis** button → triggers full pipeline
- Displays:
  - **Pipeline Path badges** (color-coded nodes visited)
  - **Clinical Answer** box (dark text on light background)
  - **Source chips** (which documents were used)
  - **Grounded / Not Grounded** verdict with confidence %
  - **Document Grading expander** (each doc's relevance label + reason)
  - **Live Metrics** (4 colored cards)

### Tab 2: Metrics Dashboard
- Aggregate averages across the session
- Per-query breakdown table
- CSV download button

### Tab 3: Session History
- Expandable cards for every past query
- Shows pipeline path, answer excerpt, and all 4 metrics

### Tab 4: Architecture
- SVG diagram of the full CRAG pipeline
- Metric formulas reference
- Tech stack summary

### Sidebar
- Groq API key input (auto-loaded from `.env`)
- Tavily API key input (auto-loaded from `.env`)
- PDF upload → indexes into ChromaDB dynamically
- k-docs retrieval slider (3-10)
- Session history with clear button

---

## 10. 📁 Project Files Breakdown {#files}

| File | Role |
|---|---|
| `Notebook/medical_crag.ipynb` | Main development notebook — full CRAG pipeline, evaluation |
| `streamlit_app.py` | Production Streamlit frontend (952 lines) |
| `ingest.py` | Standalone PDF ingestion script — builds ChromaDB |
| `.env` | API keys (Groq, Tavily, LangSmith) — never committed to git |
| `requirements.txt` | All Python dependencies |
| `chroma_medical_db/` | Persisted ChromaDB vector store (built by ingest.py) |
| `data/` | Source medical PDF files (e.g., Medical_book.pdf — 637 pages) |
| `crag_pipeline.svg` | Architecture diagram shown in the Streamlit UI |
| `crag_evaluation_results.csv` | Per-question Naive RAG vs CRAG comparison |
| `crag_summary_metrics.csv` | Aggregated metric summary |
| `medibot/` | Python virtual environment folder |

---

## 11. ✅ Key Advantages vs Naive RAG {#advantages}

```
NAIVE RAG:                           CRAG (This Project):
──────────                           ────────────────────
Retrieve → Generate                  Retrieve → Grade → Route → (Transform/WebSearch) → Generate → Verify

Problems:                            Solutions:
❌ No quality check on docs          ✅ LLM-as-judge grades all docs
❌ Irrelevant docs → bad answers     ✅ Irrelevant query → rewritten with clinical terms
❌ No web fallback                   ✅ Tavily web search when local KB fails
❌ Hallucinations not detected       ✅ Hallucination check before returning answer
❌ No retry mechanism                ✅ Circuit breaker (retry_count limit)
❌ No observability                  ✅ LangSmith traces every call
❌ Static pipeline                   ✅ LangGraph stateful conditional flow
```

### Typical Performance Improvement (from evaluation results):
- **Hallucination Rate**: CRAG reduces by ~60-70%
- **Faithfulness**: CRAG improves by ~30-40%
- **Answer Relevance**: Comparable (both good when relevant docs exist)
- **Recall@5**: CRAG improves when query transformation kicks in

---

## Summary — What This Project Demonstrates

| Skill Area | What Was Used |
|---|---|
| **Generative AI** | Groq Llama 3.1 8B for generation, grading, verification |
| **RAG Systems** | Retrieval-Augmented Generation with ChromaDB + embeddings |
| **Agentic AI** | LangGraph stateful multi-node pipeline with conditional edges |
| **Vector Databases** | ChromaDB with 384-dim sentence transformer embeddings |
| **NLP** | Semantic similarity, text chunking, query transformation |
| **LLM Evaluation** | LLM-as-judge, hallucination detection, faithfulness scoring |
| **MLOps/Observability** | LangSmith tracing, real-time metric tracking |
| **Full-Stack AI** | Streamlit frontend with PDF upload, history, and dashboards |
| **Document Processing** | PyPDF, RecursiveCharacterTextSplitter, metadata extraction |
| **API Integration** | Groq API, Tavily Search API, LangSmith API |
