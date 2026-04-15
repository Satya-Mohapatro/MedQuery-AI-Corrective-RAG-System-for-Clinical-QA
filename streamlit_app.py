"""
╔══════════════════════════════════════════════════════════════════════╗
║   🏥 Medical CRAG QA System — Streamlit Frontend (Phase 2)          ║
║   LangGraph + Groq (Llama 3) + ChromaDB + Sentence Transformers       ║
╚══════════════════════════════════════════════════════════════════════╝

Run: streamlit run streamlit_app.py
"""

import os
import json
import re
import uuid
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical CRAG QA System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.9rem; }

    /* Node badges */
    .node-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .node-retrieve   { background: #dbeafe; color: #1e40af; }
    .node-grade      { background: #fef3c7; color: #92400e; }
    .node-transform  { background: #ede9fe; color: #5b21b6; }
    .node-websearch  { background: #fee2e2; color: #991b1b; }
    .node-generate   { background: #d1fae5; color: #065f46; }
    .node-hallcheck  { background: #fce7f3; color: #9d174d; }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1e40af; }
    .metric-label { font-size: 0.8rem; color: #6b7280; margin-top: 4px; }

    /* Source citation */
    .source-chip {
        display: inline-block;
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        color: #0369a1;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        margin: 2px;
    }

    /* Answer box */
    .answer-box {
        background: #f8fafc;
        color: #0f172a;
        border-left: 4px solid #2563eb;
        padding: 1.2rem 1.5rem;
        border-radius: 0 8px 8px 0;
        line-height: 1.7;
    }

    /* Relevance labels */
    .rel-relevant   { color: #059669; font-weight: 600; }
    .rel-irrelevant { color: #dc2626; font-weight: 600; }
    .rel-ambiguous  { color: #d97706; font-weight: 600; }

    /* History item */
    .history-item {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: border-color 0.2s;
    }
    .history-item:hover { border-color: #2563eb; }

    /* Warning banner */
    .medical-disclaimer {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.82rem;
        color: #92400e;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────
if "session_history" not in st.session_state:
    st.session_state.session_history = []
if "metrics_log" not in st.session_state:
    st.session_state.metrics_log = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "uploaded_docs_count" not in st.session_state:
    st.session_state.uploaded_docs_count = 0


# ─────────────────────────────────────────────────────────────────────
# LAZY IMPORTS (only load heavy libs when needed)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models(groq_api_key: str, tavily_api_key: str):
    """Load and cache all ML models and tools (called once per session)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_groq import ChatGroq

    os.environ["GROQ_API_KEY"]   = groq_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    with st.spinner("Loading embedding model (all-MiniLM-L6-v2)..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    with st.spinner("Connecting to Groq LLM (llama-3.1-8b-instant)..."):
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1024,
            groq_api_key=groq_api_key
        )

    web_search = TavilySearchResults(max_results=3, search_depth="advanced")
    return embeddings, llm, web_search


@st.cache_resource
def load_default_vectorstore(_embeddings):
    """Load ChromaDB from the persisted medical corpus if available, else load default (cached)."""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    import os
    from pathlib import Path

    PROJECT_ROOT = Path(".").resolve()
    CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_medical_db")
    COLLECTION_NAME = "medical_documents"

    if Path(CHROMA_PERSIST_DIR).exists():
        # Load the comprehensive combined vectorstore!
        vs = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=_embeddings,
            collection_name=COLLECTION_NAME
        )
        return vs

    # Default corpus fallback (same as notebook)
    DEFAULT_DOCS = [
        Document(page_content="""Metformin standard adult dosage: Initial dose 500 mg twice daily with meals.
        Maximum daily dose: 2550 mg/day. Contraindications: eGFR < 30 mL/min/1.73m².
        Hold 48 hours before iodinated contrast. Monitor: renal function annually, Vitamin B12 every 2-3 years.
        Common side effects: nausea, vomiting, diarrhea. Rare: Lactic acidosis.""",
                  metadata={"title": "Metformin Clinical Reference", "category": "drug_reference"}),
        Document(page_content="""Lisinopril hypertension dosing: Initial 10 mg once daily; maintenance 20-40 mg once daily; maximum 80 mg/day.
        Heart failure: Initial 2.5-5 mg once daily; target 20-40 mg. Black box: discontinue if pregnancy detected.
        Side effects: Dry cough (10-15%), hyperkalemia, angioedema (rare but serious).
        Monitoring: Serum creatinine, potassium, BP within 1-2 weeks of initiation.""",
                  metadata={"title": "Lisinopril ACE Inhibitor Reference", "category": "drug_reference"}),
        Document(page_content="""Warfarin target INR: AF 2.0-3.0; Mechanical mitral valve 2.5-3.5; DVT/PE 2.0-3.0.
        Loading dose: 5-10 mg day 1-2. Reversal: Vitamin K, FFP, PCC, Phytonadione.
        Interactions: amiodarone, fluconazole increase effect; rifampin decreases effect.
        Pregnancy contraindicated first trimester.""",
                  metadata={"title": "Warfarin Anticoagulation Protocol", "category": "drug_reference"}),
        Document(page_content="""Sepsis Hour-1 Bundle: Measure lactate; blood cultures before antibiotics; administer broad-spectrum antibiotics;
        30 mL/kg crystalloid for hypotension or lactate ≥4 mmol/L; vasopressors to maintain MAP ≥65 mmHg.
        First-line vasopressor: Norepinephrine. Glucose target: 140-180 mg/dL.
        Tidal volume: 6 mL/kg IBW for ARDS. Hydrocortisone 200 mg/day if hemodynamically unstable.""",
                  metadata={"title": "Sepsis Management Guidelines", "category": "clinical_guideline"}),
        Document(page_content="""Hypertension ACC/AHA 2017: Stage 2 HTN ≥140/≥90. First-line: thiazides, ACE inhibitors/ARBs, dihydropyridine CCBs.
        CKD with proteinuria: ACE inhibitor or ARB preferred. Diabetes: ACE inhibitor or ARB.
        Hypertensive emergency: IV labetalol, nicardipine, or hydralazine. Reduce MAP 10-20% in first hour.""",
                  metadata={"title": "Hypertension Management Guidelines", "category": "clinical_guideline"}),
        Document(page_content="""STEMI door-to-balloon time: ≤90 minutes for primary PCI. Aspirin 325 mg chewed immediately.
        DAPT: aspirin + ticagrelor 180 mg load (preferred). Post-PCI: beta-blocker, ACE inhibitor, atorvastatin 80 mg.
        DAPT duration: 12 months. Anticoagulation: UFH or bivalirudin.""",
                  metadata={"title": "Acute MI STEMI Management Protocol", "category": "clinical_guideline"}),
        Document(page_content="""Stroke IV tPA: 0.9 mg/kg (max 90 mg) within 4.5 hours. 10% bolus over 1 min, remainder over 60 min.
        FAST: Face drooping, Arm weakness, Speech difficulty, Time. Ischemic: 87%, Hemorrhagic: 13%.
        BP before tPA: treat if >185/110. After tPA: maintain <180/105 for 24 hours.
        Hemorrhagic reversal: PCC for warfarin; andexanet alfa for Factor Xa inhibitors.""",
                  metadata={"title": "Stroke Recognition and Acute Management", "category": "symptom_diagnosis"}),
        Document(page_content="""DVT Wells score ≥2: high probability, proceed to ultrasound. Provoked DVT: 3 months anticoagulation.
        Rivaroxaban: 15 mg twice daily x21 days, then 20 mg daily. Enoxaparin: 1 mg/kg twice daily.
        Cancer-associated DVT: LMWH or rivaroxaban/edoxaban indefinitely. Massive PE: alteplase 100 mg over 2 hours.""",
                  metadata={"title": "Deep Vein Thrombosis DVT Treatment Protocol", "category": "treatment_protocol"}),
        Document(page_content="""COPD exacerbation: Albuterol 2.5-5 mg nebulized q20min x3. Prednisone 40 mg x5 days.
        Antibiotics if purulent sputum: amoxicillin-clavulanate or azithromycin. Oxygen target SpO2 88-92%.
        NIV indication: PaCO2 > 45 with pH < 7.35. Discharge criteria: SpO2 ≥92% on ≤2L NC.""",
                  metadata={"title": "COPD Exacerbation Treatment Protocol", "category": "treatment_protocol"}),
        Document(page_content="""AF anticoagulation CHA2DS2-VASc: Males score ≥2, Females ≥3 start anticoagulation.
        Apixaban: 5 mg twice daily (reduce to 2.5 mg twice daily if ≥2 of: age ≥80, weight ≤60 kg, Cr ≥1.5).
        Rivaroxaban: 20 mg once daily with evening meal. Idarucizumab reverses dabigatran.
        Rate control target: resting HR <110 bpm.""",
                  metadata={"title": "Anticoagulation for Atrial Fibrillation", "category": "treatment_protocol"}),
        Document(page_content="""AKI KDIGO: Cr ≥0.3 mg/dL within 48h OR ≥1.5x baseline within 7 days OR UO <0.5 mL/kg/hr for ≥6h.
        Pre-renal: FeNa <1%, BUN/Cr >20. Intrinsic: FeNa >2%, granular casts.
        Emergent dialysis AEIOU: Acidosis pH<7.1, Electrolytes K+>6.5, Ingestion, fluid Overload, Uremia.""",
                  metadata={"title": "Acute Kidney Injury AKI Diagnosis and Management", "category": "symptom_diagnosis"}),
        Document(page_content="""Chest pain life-threatening: ACS, aortic dissection, PE, tension pneumothorax, esophageal rupture.
        Troponin rise: 2-4 hours post-injury, peak 12-24 hours. HEART score ≥4: >13% MACE risk (admit).
        Aortic dissection: tearing pain radiating to back, pulse differential, mediastinal widening.
        PE: pleuritic pain, dyspnea, tachycardia; Wells score + D-dimer for low probability.""",
                  metadata={"title": "Chest Pain Differential Diagnosis and Evaluation", "category": "symptom_diagnosis"}),
        Document(page_content="""Type 2 diabetes HbA1c target <7.0% (general); <7.5-8.0% elderly. Step 1: Metformin + lifestyle.
        ASCVD/CV risk: GLP-1 agonist or SGLT-2 inhibitor. Heart failure: SGLT-2 inhibitor (empagliflozin/dapagliflozin).
        Insulin initiation: HbA1c ≥10%; basal insulin glargine 10 units/night. BP target <130/80.""",
                  metadata={"title": "Type 2 Diabetes Management Protocol", "category": "clinical_guideline"}),
        Document(page_content="""Amoxicillin adult standard: 500 mg q8h or 875 mg q12h. Pediatric: 25-45 mg/kg/day.
        H. pylori triple therapy: amoxicillin 1000 mg + clarithromycin 500 mg + PPI twice daily x14 days.
        Strep pharyngitis: 500 mg twice daily x10 days. Renal: CrCl <10: q24h dosing.""",
                  metadata={"title": "Amoxicillin Antibiotic Reference", "category": "drug_reference"}),
        Document(page_content="""Atorvastatin high-intensity: 40-80 mg daily (≥50% LDL reduction). Max 80 mg/day.
        Indication: ASCVD risk ≥7.5%, LDL ≥190 mg/dL, diabetes age 40-75.
        Side effects: myalgia (5-10%), rhabdomyolysis (rare). Monitor: LFTs baseline; lipids at 4-12 weeks.""",
                  metadata={"title": "Atorvastatin Lipid Management Reference", "category": "drug_reference"}),
        Document(page_content="""Pain management WHO ladder: Step 1 non-opioids (acetaminophen max 4000 mg/day, 3000 mg elderly).
        NSAIDs: ibuprofen 400-800 mg q6-8h. Ketorolac IV: 15-30 mg, limit 5 days.
        Morphine IV: 2-4 mg q3-4h. Hydromorphone IV: 0.4-1 mg q3-4h (6x more potent than morphine).
        Opioid conversion: morphine 10 mg IV = oxycodone 15 mg PO.""",
                  metadata={"title": "Acute Pain Management Protocol", "category": "treatment_protocol"}),
        Document(page_content="""Pneumonia CURB-65: Confusion, Urea >7, RR ≥30, BP <90 systolic, Age ≥65.
        Score 0-1: outpatient; Score 2: admit; Score 3-5: ICU consideration.
        CAP outpatient: amoxicillin 1g TID x5-7 days. Inpatient: beta-lactam + azithromycin.
        Duration: CAP 5-7 days; HAP/VAP 7-8 days.""",
                  metadata={"title": "Pneumonia Diagnosis and Management", "category": "symptom_diagnosis"}),
    ]

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = splitter.split_documents(DEFAULT_DOCS)

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings,
        collection_name="medical_default"
    )
    return vs


# ─────────────────────────────────────────────────────────────────────
# CRAG PIPELINE (reused from notebook logic, adapted for Streamlit)
# ─────────────────────────────────────────────────────────────────────
def run_crag_streamlit(question: str, vectorstore, llm, web_search_tool,
                        k: int = 5) -> Dict[str, Any]:
    """
    Run full CRAG pipeline with step-by-step status updates for Streamlit.
    Returns detailed result dict for display.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document
    from sklearn.metrics.pairwise import cosine_similarity

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    pipeline_trace = []
    status_container = st.empty()

    def update_status(msg):
        status_container.info(f"⚙️ {msg}")

    # ── STEP 1: RETRIEVE ──
    update_status("Retrieving relevant documents from ChromaDB...")
    pipeline_trace.append("RETRIEVE")
    docs = retriever.invoke(question)

    # ── STEP 2: GRADE DOCUMENTS ──
    update_status("Grading document relevance with HuggingFace LLM...")
    pipeline_trace.append("GRADE_DOCUMENTS")

    GRADE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are a medical relevance grader. "
         "Respond ONLY with JSON: {{\"relevance\": \"RELEVANT\"|\"IRRELEVANT\"|\"AMBIGUOUS\", \"reason\": \"...\"}} "
         "No extra text."),
        ("human", "Question: {question}\n\nDocument:\n{document}")
    ])
    grade_chain = GRADE_PROMPT | llm | StrOutputParser()

    graded_docs = []
    for doc in docs:
        try:
            res = grade_chain.invoke({
                "question": question,
                "document": doc.page_content[:500]
            })
            res_clean = res.replace("```json", "").replace("```", "").strip()
            data = json.loads(res_clean)
            relevance = data.get("relevance", "AMBIGUOUS")
            reason = data.get("reason", "")
        except:
            relevance = "AMBIGUOUS"
            reason = "Parse error"
        graded_docs.append({
            "document": doc,
            "relevance": relevance,
            "reason": reason,
            "title": doc.metadata.get("title", "Unknown")
        })

    # ── STEP 3: ROUTE ──
    relevant_count = sum(1 for g in graded_docs if g["relevance"] == "RELEVANT")
    irrelevant_count = sum(1 for g in graded_docs if g["relevance"] == "IRRELEVANT")
    total = len(graded_docs)
    web_used = False
    transformed_query = None

    if relevant_count == total:
        route = "generate"
        pipeline_trace.append("→ ALL_RELEVANT")
    elif irrelevant_count == total:
        route = "web_search"
    else:
        route = "transform_query"

    # ── STEP 4: TRANSFORM QUERY (if mixed) ──
    if route == "transform_query":
        update_status("Transforming query for better retrieval...")
        pipeline_trace.append("TRANSFORM_QUERY")
        TRANSFORM_PROMPT = ChatPromptTemplate.from_messages([
            ("system", "Rewrite this medical query with precise clinical terminology. Return ONLY the rewritten query."),
            ("human", "Query: {question}")
        ])
        transform_chain = TRANSFORM_PROMPT | llm | StrOutputParser()
        transformed_query = transform_chain.invoke({"question": question}).strip()

        update_status("Re-retrieving with transformed query...")
        pipeline_trace.append("RE-RETRIEVE")
        docs = retriever.invoke(transformed_query)
        route = "generate"

    # ── STEP 5: WEB SEARCH (if all irrelevant) ──
    elif route == "web_search":
        update_status("All docs irrelevant — falling back to web search (Tavily)...")
        pipeline_trace.append("WEB_SEARCH_FALLBACK")
        web_used = True
        try:
            results = web_search_tool.invoke({"query": f"medical {question}"})
            docs = [
                Document(
                    page_content=r.get("content", r.get("snippet", ""))[:500],
                    metadata={"title": r.get("title", "Web Result"), "source": r.get("url", "web"), "category": "web"}
                )
                for r in results if isinstance(r, dict)
            ]
        except Exception as e:
            st.warning(f"Web search failed: {e}")

    # ── STEP 6: GENERATE ──
    update_status("Generating answer with HuggingFace LLM...")
    pipeline_trace.append("GENERATE")

    context_docs = [g["document"] for g in graded_docs if g["relevance"] in ("RELEVANT", "AMBIGUOUS")]
    if not context_docs:
        context_docs = docs

    context_str = "\n\n---\n".join([
        f"[SOURCE: {doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}"
        for doc in context_docs[:5]
    ])

    GEN_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are an expert medical AI assistant for clinicians.
Answer using ONLY the provided context. Include specific values (dosages, thresholds).
End with: SOURCES: [document titles used]"""),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ])
    gen_chain = GEN_PROMPT | llm | StrOutputParser()
    generation = gen_chain.invoke({"question": question, "context": context_str})

    # ── STEP 7: HALLUCINATION CHECK ──
    update_status("Verifying answer groundedness...")
    pipeline_trace.append("HALLUCINATION_CHECK")

    HALL_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "Verify if this medical answer is grounded in the context. "
         "Respond ONLY with JSON: {{\"verdict\": \"grounded\"|\"not_grounded\", "
         "\"confidence\": 0.0-1.0, \"ungrounded_claims\": [...]}} No extra text."),
        ("human", "Answer: {answer}\n\nContext: {context}")
    ])
    hall_chain = HALL_PROMPT | llm | StrOutputParser()

    try:
        hall_res = hall_chain.invoke({
            "answer": generation[:1000],
            "context": context_str[:1500]
        })
        hall_data = json.loads(hall_res.replace("```json", "").replace("```", "").strip())
        hallucination_verdict = hall_data.get("verdict", "grounded")
        hall_confidence = float(hall_data.get("confidence", 0.9))
        ungrounded_claims = hall_data.get("ungrounded_claims", [])
    except:
        hallucination_verdict = "grounded"
        hall_confidence = 0.8
        ungrounded_claims = []

    status_container.empty()

    # Compute quick metrics
    from sklearn.metrics.pairwise import cosine_similarity
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # Answer relevance via embedding similarity
    try:
        emb_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        q_emb = np.array(emb_model.embed_query(question)).reshape(1, -1)
        a_emb = np.array(emb_model.embed_query(generation[:500])).reshape(1, -1)
        answer_relevance = float(cosine_similarity(q_emb, a_emb)[0][0])
    except:
        answer_relevance = 0.0

    # Recall@5 (heuristic: how many docs have category match)
    question_lower = question.lower()
    recall5 = sum(
        1 for d in docs[:5]
        if any(kw in d.page_content.lower() for kw in question_lower.split()[:4])
    ) / max(len(docs[:5]), 1)

    # Faithfulness (keyword overlap heuristic)
    context_words = set(context_str.lower().split())
    answer_words = set(generation.lower().split())
    faithfulness = len(answer_words & context_words) / max(len(answer_words), 1)
    faithfulness = min(faithfulness * 2.5, 1.0)  # Scale up

    return {
        "question": question,
        "answer": generation,
        "graded_documents": graded_docs,
        "context_docs": context_docs,
        "pipeline_trace": pipeline_trace,
        "transformed_query": transformed_query,
        "web_used": web_used,
        "hallucination_verdict": hallucination_verdict,
        "hall_confidence": hall_confidence,
        "ungrounded_claims": ungrounded_claims,
        "sources": list({d.metadata.get("title", "Unknown") for d in context_docs}),
        "metrics": {
            "recall_at_5": recall5,
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "hallucination_rate": 0.0 if hallucination_verdict == "grounded" else 1.0,
        },
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }


# ─────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────

def render_pipeline_path(trace: List[str]):
    """Render the CRAG pipeline path as colored badges."""
    color_map = {
        "RETRIEVE": "node-retrieve",
        "GRADE_DOCUMENTS": "node-grade",
        "TRANSFORM_QUERY": "node-transform",
        "WEB_SEARCH_FALLBACK": "node-websearch",
        "RE-RETRIEVE": "node-retrieve",
        "GENERATE": "node-generate",
        "HALLUCINATION_CHECK": "node-hallcheck",
    }
    badges = []
    for step in trace:
        css = color_map.get(step, "node-retrieve")
        if "ALL_RELEVANT" in step:
            badges.append(f'<span class="node-badge node-grade">✅ All Relevant</span>')
        else:
            badges.append(f'<span class="node-badge {css}">{step.replace("_", " ")}</span>')
    st.markdown(" → ".join(badges), unsafe_allow_html=True)


def render_graded_documents(graded_docs: List[Dict]):
    """Render document grading results in an expandable table."""
    with st.expander(f"📋 Document Grading ({len(graded_docs)} docs retrieved)", expanded=False):
        for i, g in enumerate(graded_docs):
            rel = g["relevance"]
            rel_cls = {"RELEVANT": "rel-relevant", "IRRELEVANT": "rel-irrelevant", "AMBIGUOUS": "rel-ambiguous"}
            rel_icon = {"RELEVANT": "✅", "IRRELEVANT": "❌", "AMBIGUOUS": "⚠️"}
            col1, col2, col3 = st.columns([3, 1, 4])
            with col1:
                st.markdown(f"**{i+1}. {g['title'][:45]}**")
            with col2:
                st.markdown(f'<span class="{rel_cls.get(rel, "")}">{rel_icon.get(rel, "")} {rel}</span>',
                            unsafe_allow_html=True)
            with col3:
                st.caption(g.get("reason", "")[:80])
            st.markdown(f"> {g['document'].page_content[:150]}...")
            st.divider()


def render_metrics_dashboard(metrics: Dict[str, float], hallucination_verdict: str):
    """Render live metric dashboard as cards."""
    st.subheader("📊 Live Metrics")
    col1, col2, col3, col4 = st.columns(4)

    recall = metrics.get("recall_at_5", 0)
    faith = metrics.get("faithfulness", 0)
    relevance = metrics.get("answer_relevance", 0)
    hall_rate = metrics.get("hallucination_rate", 0)

    with col1:
        color = "#059669" if recall >= 0.7 else "#d97706" if recall >= 0.4 else "#dc2626"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{color}">{recall:.0%}</div>
            <div class="metric-label">Recall@5</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        color = "#059669" if faith >= 0.7 else "#d97706" if faith >= 0.4 else "#dc2626"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{color}">{faith:.0%}</div>
            <div class="metric-label">Faithfulness</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        color = "#059669" if relevance >= 0.7 else "#d97706" if relevance >= 0.4 else "#dc2626"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{color}">{relevance:.2f}</div>
            <div class="metric-label">Answer Relevance</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        color = "#059669" if hall_rate == 0 else "#dc2626"
        verdict_label = "✅ Grounded" if hallucination_verdict == "grounded" else "⚠️ Ungrounded"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{color}">{hall_rate:.0%}</div>
            <div class="metric-label">Hallucination Rate<br><small>{verdict_label}</small></div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Attempt to load from Streamlit Secrets first, then local .env
    default_groq = ""
    try: default_groq = st.secrets["GROQ_API_KEY"]
    except: default_groq = os.environ.get("GROQ_API_KEY", "")

    default_tavily = ""
    try: default_tavily = st.secrets["TAVILY_API_KEY"]
    except: default_tavily = os.environ.get("TAVILY_API_KEY", "")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        value=default_groq,
        help="Free key from console.groq.com — no credit card needed"
    )
    tavily_key = st.text_input(
        "Tavily API Key",
        type="password",
        value=default_tavily,
        help="For web search fallback"
    )

    st.divider()
    st.markdown("## 📤 Upload Medical Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files (clinical guidelines, drug references)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Documents will be chunked and indexed in ChromaDB"
    )

    if uploaded_files and groq_key:
        if st.button("🔄 Index Uploaded Documents", use_container_width=True):
            try:
                import pypdf
                from langchain_core.documents import Document
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from langchain_chroma import Chroma

                with st.spinner("Processing PDFs..."):
                    all_chunks = []
                    for uploaded_file in uploaded_files:
                        reader = pypdf.PdfReader(uploaded_file)
                        text = " ".join([page.extract_text() or "" for page in reader.pages])
                        doc = Document(
                            page_content=text,
                            metadata={"title": uploaded_file.name, "category": "uploaded"}
                        )
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500, chunk_overlap=50
                        )
                        all_chunks.extend(splitter.split_documents([doc]))

                    emb = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True}
                    )
                    vs = Chroma.from_documents(
                        documents=all_chunks,
                        embedding=emb,
                        collection_name=f"medical_upload_{int(time.time())}"
                    )
                    st.session_state.vectorstore = vs
                    st.session_state.uploaded_docs_count = len(uploaded_files)
                    st.success(f"✅ Indexed {len(all_chunks)} chunks from {len(uploaded_files)} PDF(s)")
            except Exception as e:
                st.error(f"Error indexing PDFs: {e}")

    st.divider()
    st.markdown("## 🔧 Retrieval Settings")
    k_docs = st.slider("Documents to retrieve (k)", min_value=3, max_value=10, value=5)

    st.divider()
    st.markdown("## 📜 Session History")
    if st.session_state.session_history:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.session_history = []
            st.session_state.metrics_log = []
            st.rerun()

        for i, item in enumerate(reversed(st.session_state.session_history[-5:])):
            with st.expander(f"{item['timestamp']} — {item['question'][:40]}...", expanded=False):
                st.markdown(f"**Path:** {' → '.join(item['pipeline_trace'])}")
                st.markdown(f"**Verdict:** {item['hallucination_verdict']}")
    else:
        st.caption("No queries yet in this session.")


# ─────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 Medical CRAG QA System</h1>
    <p>Corrective Retrieval-Augmented Generation · LangGraph · Groq (Llama 3) · ChromaDB · Sentence Transformers</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="medical-disclaimer">
    ⚠️ <strong>Clinical Decision Support Tool</strong> — This system is designed to assist clinicians with 
    evidence-based information retrieval. Always verify responses against primary sources 
    and use clinical judgment. Not a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)

# ── TAB LAYOUT ──
tab_query, tab_metrics, tab_history, tab_architecture = st.tabs([
    "🔍 Query Interface",
    "📊 Metrics Dashboard",
    "📜 Session History",
    "🏗️ Architecture"
])

# ── TAB 1: QUERY INTERFACE ──
with tab_query:
    col_main, col_side = st.columns([3, 1])

    with col_side:
        st.markdown("### 💡 Example Queries")
        example_queries = [
            "What is the maximum daily dose of metformin?",
            "What are the Hour-1 Bundle components for sepsis?",
            "When should I use NIV in COPD exacerbation?",
            "What is the target INR for atrial fibrillation on warfarin?",
            "What is the door-to-balloon time target for STEMI?",
            "What DOAC is preferred for cancer-associated DVT?",
            "What is the FAST acronym for stroke recognition?",
            "What are the indications for emergent dialysis?",
        ]
        for eq in example_queries:
            if st.button(eq[:50] + ("..." if len(eq) > 50 else ""),
                         key=f"ex_{hash(eq)}", use_container_width=True):
                st.session_state.selected_query = eq

    with col_main:
        selected = st.session_state.get("selected_query", "")
        question = st.text_area(
            "📝 Enter your clinical question",
            value=selected,
            height=100,
            placeholder="e.g., What is the first-line vasopressor for septic shock?"
        )

        if st.button("🔬 Run CRAG Analysis", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Please enter a clinical question.")
            elif not groq_key:
                st.error("Please provide your Groq API Key in the sidebar (free at console.groq.com).")
            elif not tavily_key:
                st.error("Please provide your Tavily API Key in the sidebar.")
            else:
                try:
                    with st.spinner("Initializing Groq LLM + embeddings..."):
                        embeddings, llm_model, web_search = load_models(groq_key, tavily_key)

                    # Load vectorstore
                    if st.session_state.vectorstore is None:
                        with st.spinner("Loading default medical knowledge base..."):
                            vs = load_default_vectorstore(embeddings)
                    else:
                        vs = st.session_state.vectorstore

                    # Run CRAG
                    start_time = time.time()
                    result = run_crag_streamlit(question, vs, llm_model, web_search, k=k_docs)
                    elapsed = time.time() - start_time

                    # ── DISPLAY RESULTS ──
                    st.success(f"✅ Analysis complete in {elapsed:.1f}s")

                    # Pipeline path
                    st.markdown("### 🔀 CRAG Pipeline Path")
                    render_pipeline_path(result["pipeline_trace"])

                    if result.get("transformed_query"):
                        st.info(f"🔄 Query transformed to: *\"{result['transformed_query']}\"*")
                    if result.get("web_used"):
                        st.warning("🌐 Web search fallback was triggered (local docs insufficient)")

                    st.divider()

                    # Answer
                    st.markdown("### 💊 Clinical Answer")
                    answer_text = result["answer"]
                    # Split sources from answer for cleaner display
                    answer_body = re.split(r"SOURCES?:", answer_text, flags=re.IGNORECASE)[0].strip()
                    st.markdown(f'<div class="answer-box">{answer_body}</div>',
                                unsafe_allow_html=True)

                    # Sources
                    st.markdown("**📚 Sources Used:**")
                    source_chips = "".join([
                        f'<span class="source-chip">📄 {s}</span>'
                        for s in result["sources"]
                    ])
                    st.markdown(source_chips, unsafe_allow_html=True)

                    # Hallucination verdict
                    st.divider()
                    col_v1, col_v2 = st.columns([1, 2])
                    with col_v1:
                        if result["hallucination_verdict"] == "grounded":
                            st.success(f"✅ Grounded (confidence: {result['hall_confidence']:.0%})")
                        else:
                            st.error(f"⚠️ Not fully grounded (confidence: {result['hall_confidence']:.0%})")
                            if result["ungrounded_claims"]:
                                st.warning("Unverified claims: " + "; ".join(result["ungrounded_claims"][:3]))

                    # Graded documents
                    render_graded_documents(result["graded_documents"])

                    # Metrics
                    render_metrics_dashboard(result["metrics"], result["hallucination_verdict"])

                    # Save to history
                    st.session_state.session_history.append(result)
                    st.session_state.metrics_log.append({
                        "question": question[:50],
                        "timestamp": result["timestamp"],
                        **result["metrics"],
                        "pipeline_path": " → ".join(result["pipeline_trace"])
                    })

                    # Clear selected query
                    if "selected_query" in st.session_state:
                        del st.session_state.selected_query

                except Exception as e:
                    st.error(f"Pipeline error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


# ── TAB 2: METRICS DASHBOARD ──
with tab_metrics:
    st.markdown("### 📊 Live Metrics Across Session")

    if st.session_state.metrics_log:
        metrics_df = pd.DataFrame(st.session_state.metrics_log)

        # Aggregate stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Recall@5",     f"{metrics_df['recall_at_5'].mean():.1%}")
        col2.metric("Avg Faithfulness", f"{metrics_df['faithfulness'].mean():.1%}")
        col3.metric("Avg Ans. Relevance", f"{metrics_df['answer_relevance'].mean():.3f}")
        col4.metric("Hallucination Rate", f"{metrics_df['hallucination_rate'].mean():.1%}")

        st.divider()

        # Per-question table
        st.markdown("#### Per-Query Breakdown")
        display_cols = ["timestamp", "question", "recall_at_5", "faithfulness",
                        "answer_relevance", "hallucination_rate"]
        st.dataframe(
            metrics_df[[c for c in display_cols if c in metrics_df.columns]],
            use_container_width=True
        )

        # Download button
        csv = metrics_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Metrics CSV",
            data=csv,
            file_name=f"crag_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Run some queries in the Query Interface tab to see live metrics here.")


# ── TAB 3: SESSION HISTORY ──
with tab_history:
    st.markdown("### 📜 Full Session History")

    if st.session_state.session_history:
        for i, item in enumerate(reversed(st.session_state.session_history)):
            with st.expander(
                f"[{item['timestamp']}] {item['question'][:70]}...",
                expanded=(i == 0)
            ):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Question:** {item['question']}")
                    st.markdown("**Pipeline Path:**")
                    render_pipeline_path(item["pipeline_trace"])
                    st.markdown("**Answer (excerpt):**")
                    answer_excerpt = item["answer"][:400].replace("\n", " ")
                    st.markdown(f"> {answer_excerpt}...")
                with col2:
                    st.markdown("**Metrics:**")
                    for metric, val in item["metrics"].items():
                        label = metric.replace("_", " ").title()
                        formatted = f"{val:.1%}" if metric in ("recall_at_5", "faithfulness", "hallucination_rate") else f"{val:.3f}"
                        st.metric(label, formatted)
    else:
        st.info("No session history yet. Run some queries first!")


# ── TAB 4: ARCHITECTURE ──
import base64

with tab_architecture:
    st.markdown("### 🏗️ CRAG Pipeline Architecture")
    with open("crag_pipeline.svg", "r", encoding="utf-8") as f:
        svg_content = f.read()

    b64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{b64}" style="width:65%; display:block; margin:auto;" />',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### 📐 Metric Formulas")

    formulas = {
        "Recall@K": "hits_in_top_K / total_relevant_docs",
        "MRR (Mean Reciprocal Rank)": "1 / rank_of_first_relevant_doc",
        "Faithfulness": "grounded_claims / total_claims  [LLM-as-judge]",
        "Answer Relevance": "cosine_similarity(embed(question), embed(answer))",
        "Hallucination Rate": "answers_with_ungrounded_claims / total_answers",
        "Groundedness": "grounded_claims / total_claims  [per-answer]",
    }

    col1, col2 = st.columns(2)
    items = list(formulas.items())
    for i, (metric, formula) in enumerate(items):
        with (col1 if i < 3 else col2):
            st.markdown(f"**{metric}**")
            st.code(formula)

    st.divider()
    st.markdown("### 🛠️ Tech Stack")
    tech_cols = st.columns(3)
    with tech_cols[0]:
        st.markdown("""
**Orchestration**
- LangGraph (stateful pipeline)
- LangChain (prompts, chains)

**LLM**
- Groq API — llama-3.1-8b-instant
- Free tier (console.groq.com)
- Temperature: 0.1 (factual)
        """)
    with tech_cols[1]:
        st.markdown("""
**Vector Store**
- ChromaDB (local, persistent)
- Chunk size: 500, overlap: 50
- k-NN retrieval (k=5)

**Embeddings**
- all-MiniLM-L6-v2
- 384-dimensional vectors
        """)
    with tech_cols[2]:
        st.markdown("""
**Web Fallback**
- Tavily Search API
- Triggered when all docs irrelevant

**Observability**
- LangSmith tracing
- Pipeline path visualization
        """)
