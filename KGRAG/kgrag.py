import os
import requests
import streamlit as st
import networkx as nx
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Tuple

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# =============================================================================
# 1. Pydantic Scheme Definition & Ingestion Pipeline
# =============================================================================
class WelfareScheme(BaseModel):
    name: str = Field(description="Name of the government welfare scheme")
    department: str = Field(description="Department administering the scheme")
    target_group: str = Field(description="Demographic or beneficiary target group")
    income_limit: str = Field(description="Annual parental/family income ceiling")
    benefits: str = Field(description="Assistance, fee concession, or stipend provided")
    eligibility_notes: str = Field(description="Additional documentation or criteria required")

# Comprehensive structured records for TN Welfare Schemes (dep_id=2 and related)
DEFAULT_SCHEMES: List[Dict[str, str]] = [
    {
        "name": "Post-Matric Scholarship Scheme",
        "department": "Backward Classes Welfare Department",
        "target_group": "Post-matric students (XI, XII, Diploma, UG, PG)",
        "income_limit": "Rs. 2.5 Lakhs per annum",
        "benefits": "Tuition fee waiver, non-refundable compulsory fees, and maintenance allowances",
        "eligibility_notes": "Valid Community Certificate (BC/MBC/DNC) and Income Certificate issued by Tahsildar required."
    },
    {
        "name": "Free Education Scheme (3-Year Degree UG Courses)",
        "department": "Backward Classes Welfare Department",
        "target_group": "Undergraduate general degree students (B.A., B.Sc., B.Com)",
        "income_limit": "Rs. 2.5 Lakhs per annum",
        "benefits": "Full waiver of tuition, special fees, and examination fees in Govt. and Govt-aided arts/science colleges",
        "eligibility_notes": "First generation graduate applicants are given preferential consideration."
    },
    {
        "name": "Professional Courses Tuition Assistance (Govt Quota in Private Colleges)",
        "department": "Backward Classes Welfare Department",
        "target_group": "Students pursuing B.E., B.Tech, MBBS, BDS, Agriculture through Government Quota",
        "income_limit": "Rs. 2.5 Lakhs per annum",
        "benefits": "Tuition fee concession fixed by the Government State Fee Regulatory Committee",
        "eligibility_notes": "Allotment order through single-window government centralized counseling is mandatory."
    },
    {
        "name": "Hostel Boarding Grant and Maintenance",
        "department": "Backward Classes Welfare Department",
        "target_group": "College and school hostel boarders staying in designated departmental hostels",
        "income_limit": "Rs. 2.0 Lakhs per annum",
        "benefits": "Free lodging, boarding charges, two sets of uniforms, and textbook allowances",
        "eligibility_notes": "Distance of residence from school/college must be greater than 8 km (exempt for female students)."
    },
    {
        "name": "Higher Education Special Scholarship Scheme",
        "department": "Adi Dravidar and Tribal Welfare Department",
        "target_group": "Full-time degree and postgraduate students residing in approved college hostels",
        "income_limit": "Rs. 2.5 Lakhs per annum",
        "benefits": "Lump sum annual assistance of Rs. 7,500 for UG and Rs. 8,000 for PG towards boarding and mess dues",
        "eligibility_notes": "Must be an active resident of recognized educational institution hostels."
    }
]

def fetch_web_schemes(url: str) -> List[Dict[str, str]]:
    """Attempts live HTML table parsing from the government portal, with fallback."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    extracted = []
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if table:
                rows = table.find_all("tr")[1:]
                for row in rows:
                    cols = [td.get_text(strip=True) for td in row.find_all("td")]
                    if len(cols) >= 3:
                        extracted.append({
                            "name": cols[1],
                            "department": cols[2] if len(cols) > 2 else "Backward Classes Welfare Department",
                            "target_group": "Eligible state domicile residents",
                            "income_limit": "Rs. 2.5 Lakhs per annum",
                            "benefits": cols[3] if len(cols) > 3 else "Educational and financial grant",
                            "eligibility_notes": "Verified community and income documentation"
                        })
    except Exception:
        pass
    return extracted if extracted else DEFAULT_SCHEMES

# =============================================================================
# 2. Knowledge Graph Engine (NetworkX)
# =============================================================================
def build_knowledge_graph(schemes: List[Dict[str, str]]) -> nx.DiGraph:
    """
    Constructs a directed knowledge graph with explicit semantic relations:
    - (Department) -[ADMINISTERS]-> (Scheme)
    - (Scheme) -[TARGETS]-> (TargetGroup)
    - (Scheme) -[HAS_INCOME_CEILING]-> (IncomeLimit)
    - (Scheme) -[PROVIDES]-> (Benefit)
    """
    G = nx.DiGraph()
    for s in schemes:
        name = s["name"]
        dept = s["department"]
        target = s["target_group"]
        income = s["income_limit"]
        benefit = s["benefits"]

        # Add typed nodes
        G.add_node(name, node_type="Scheme", details=s["eligibility_notes"])
        G.add_node(dept, node_type="Department")
        G.add_node(target, node_type="TargetGroup")
        G.add_node(income, node_type="IncomeLimit")
        G.add_node(benefit, node_type="Benefit")

        # Connect directional edges with explicit relations
        G.add_edge(dept, name, relation="ADMINISTERS")
        G.add_edge(name, target, relation="TARGETS")
        G.add_edge(name, income, relation="HAS_INCOME_CEILING")
        G.add_edge(name, benefit, relation="PROVIDES")

    return G

def traverse_knowledge_graph(G: nx.DiGraph, query: str) -> str:
    """Extracts 1-hop and 2-hop neighborhood facts around recognized entities."""
    query_lower = query.lower()
    discovered_triples = set()

    for node, data in G.nodes(data=True):
        node_str = str(node).lower()
        # Direct keyword match or meaningful token overlap
        if node_str in query_lower or any(token in node_str for token in query_lower.split() if len(token) > 4):
            # Outgoing relations
            for neighbor in G.neighbors(node):
                rel = G.get_edge_data(node, neighbor).get("relation", "RELATES_TO")
                discovered_triples.add(f"({node}) --[{rel}]--> ({neighbor})")
            
            # Incoming relations
            for pred in G.predecessors(node):
                rel = G.get_edge_data(pred, node).get("relation", "RELATES_TO")
                discovered_triples.add(f"({pred}) --[{rel}]--> ({node})")

    if not discovered_triples:
        return "No direct graph relationship facts found for the tokens in this query."
    return "\n".join(f"- {t}" for t in sorted(discovered_triples))

# =============================================================================
# 3. Dense Vector Store Pipeline (FAISS)
# =============================================================================
def build_faiss_vectorstore(schemes: List[Dict[str, str]], api_key: str) -> FAISS:
    """Embeds scheme details into an in-memory FAISS similarity index."""
    documents = []
    for s in schemes:
        text = (
            f"Scheme: {s['name']}\n"
            f"Department: {s['department']}\n"
            f"Beneficiary Target Group: {s['target_group']}\n"
            f"Income Ceiling: {s['income_limit']}\n"
            f"Provided Benefits: {s['benefits']}\n"
            f"Eligibility & Rules: {s['eligibility_notes']}"
        )
        documents.append(Document(page_content=text, metadata={"scheme_name": s["name"]}))

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.from_documents(documents, embeddings)

# =============================================================================
# 4. Streamlit Application & Hybrid Orchestrator
# =============================================================================
st.set_page_config(page_title="TN Welfare Schemes Graph RAG", page_icon="🏛️", layout="wide")

st.markdown("""
### 🏛️ Tamil Nadu Government Schemes - Graph RAG Assistant
This system leverages **NetworkX (Knowledge Graph)** for relationship mapping and **FAISS** for vector search.
""")

with st.sidebar:
    st.header("⚙️ Configuration")
    openai_key = "enteryour OpenAI API Key here"  # Placeholder for user input
    
    
    portal_url = st.text_input(
        "Source URL", 
        value="https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
    )
    
    if st.button("Initialize / Re-Index Pipeline", type="primary"):
        if not openai_key:
            st.error("Please provide a valid OpenAI API Key.")
        else:
            with st.spinner("Parsing data, constructing Knowledge Graph & FAISS index..."):
                raw_schemes = fetch_web_schemes(portal_url)
                st.session_state["kg"] = build_knowledge_graph(raw_schemes)
                st.session_state["faiss_index"] = build_faiss_vectorstore(raw_schemes, openai_key)
                st.session_state["scheme_count"] = len(raw_schemes)
                st.session_state["ready"] = True
                st.success(f"Successfully indexed {len(raw_schemes)} schemes into Graph & FAISS!")

    if st.session_state.get("ready"):
        st.divider()
        st.metric("Graph Nodes", st.session_state["kg"].number_of_nodes())
        st.metric("Graph Edges", st.session_state["kg"].number_of_edges())
        st.metric("Schemes in FAISS", st.session_state["scheme_count"])

# Initialize conversational state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display conversation messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Execution
if user_query := st.chat_input("Ask about scholarship schemes, income ceilings, departments..."):
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    if not openai_key:
        with st.chat_message("assistant"):
            warning = "⚠️ Please enter your OpenAI API Key in the left sidebar to activate the model."
            st.warning(warning)
            st.session_state.chat_history.append({"role": "assistant", "content": warning})
            
    elif not st.session_state.get("ready"):
        with st.chat_message("assistant"):
            warning = "⚠️ Click **Initialize / Re-Index Pipeline** in the sidebar before asking questions."
            st.warning(warning)
            st.session_state.chat_history.append({"role": "assistant", "content": warning})
            
    else:
        with st.chat_message("assistant"):
            with st.spinner("Performing Hybrid Graph Traversal + FAISS Retrieval..."):
                # 1. Knowledge Graph Structured Lookup
                graph_evidence = traverse_knowledge_graph(st.session_state["kg"], user_query)

                # 2. FAISS Dense Semantic Similarity Lookup
                retriever = st.session_state["faiss_index"].as_retriever(search_kwargs={"k": 2})
                vector_hits = retriever.invoke(user_query)
                vector_evidence = "\n\n".join([doc.page_content for doc in vector_hits])

                # 3. LangChain Generation Chain
                system_template = (
                    "You are an expert Tamil Nadu Government Schemes Welfare Officer and AI Advisor.\n"
                    "Provide accurate, actionable guidance using the provided Knowledge Graph facts and FAISS excerpts.\n\n"
                    "### Knowledge Graph Relationships:\n{graph_context}\n\n"
                    "### FAISS Semantic Evidence:\n{vector_context}\n\n"
                    "Synthesize a clear, authoritative response. If the criteria are not satisfied or information is missing, state so clearly."
                )
                
                prompt_pipeline = ChatPromptTemplate.from_messages([
                    ("system", system_template),
                    ("human", "{question}")
                ])
                
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=openai_key)
                chain = prompt_pipeline | llm | StrOutputParser()

                bot_response = chain.invoke({
                    "graph_context": graph_evidence,
                    "vector_context": vector_evidence,
                    "question": user_query
                })

                st.markdown(bot_response)

                # Collapsible debug section for transparency
                with st.expander("🔍 View Hybrid Retrieved Evidence (Graph + Vector)"):
                    st.subheader("1. NetworkX Graph Relationships (Triples)")
                    st.code(graph_evidence, language="markdown")
                    st.subheader("2. FAISS Nearest Document Chunks")
                    st.code(vector_evidence, language="markdown")

        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})