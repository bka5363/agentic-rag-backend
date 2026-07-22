import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

# --- APP INITIALIZATION ---
app = FastAPI(title="Agentic RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API MODELS ---
class ChatIn(BaseModel):
    question: str

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Are the documents relevant? 'yes' or 'no'")

class GradeHallucination(BaseModel):
    binary_score: str = Field(description="Is the answer grounded in facts? 'yes' or 'no'")

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Does the answer resolve the question? 'yes' or 'no'")

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Any]
    steps: List[str]
    retry_count: int

# --- GLOBAL CONFIG (DEEPSEEK & LOCAL EMBEDDINGS) ---
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com",
    temperature=0,
    model_kwargs={"thinking": {"type": "disabled"}}
)

search_tool = DuckDuckGoSearchRun()
VECTORSTORE = None

def get_retriever():
    global VECTORSTORE
    if not VECTORSTORE:
        # Lightweight FastEmbed implementation (uses <150MB RAM)
        embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
        client = QdrantClient(path="./qdrant_db")
        VECTORSTORE = QdrantVectorStore(
            client=client, 
            collection_name="agentic_rag_bge", 
            embedding=embeddings
        )
    return VECTORSTORE.as_retriever(search_kwargs={"k": 4})

# --- LANGGRAPH NODES & EDGES ---
def retrieve(state: GraphState):
    steps = state.get("steps", [])
    steps.append("retrieve")
    docs = get_retriever().invoke(state["question"])
    return {"documents": docs, "question": state["question"], "steps": steps, "retry_count": 0}

def generate(state: GraphState):
    steps = state.get("steps", [])
    steps.append("generate")
    retry_count = state.get("retry_count", 0) + (1 if state.get("generation") else 0)

    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Synthesize an answer based ONLY on the provided context. Cite sources.\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    generation = (prompt | llm).invoke({"context": context, "question": state["question"]})
    return {"generation": generation.content, "steps": steps, "retry_count": retry_count}

def grade_documents(state: GraphState):
    steps = state.get("steps", [])
    steps.append("grade_documents")
    grader = llm.with_structured_output(GradeDocuments)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Assess if the chunk is relevant to the question. 'yes' or 'no'."),
        ("human", "Chunk: {document}\n\nQuestion: {question}")
    ])

    filtered_docs = [doc for doc in state["documents"] if (prompt | grader).invoke({"document": doc.page_content, "question": state["question"]}).binary_score.lower() == "yes"]
    return {"documents": filtered_docs, "steps": steps}

def web_search(state: GraphState):
    steps = state.get("steps", [])
    steps.append("web_search")
    web_doc = Document(page_content=search_tool.invoke(state["question"]), metadata={"source": "web_search"})
    docs = state.get("documents", [])
    docs.append(web_doc)
    return {"documents": docs, "steps": steps}

def route_after_grade(state: GraphState) -> str:
    return "web_search" if not state["documents"] else "generate"

def route_after_generate(state: GraphState) -> str:
    if state.get("retry_count", 0) >= 2: return "useful"

    h_grader = llm.with_structured_output(GradeHallucination)
    h_prompt = ChatPromptTemplate.from_messages([
        ("system", "Is the response grounded in the facts? 'yes' or 'no'."),
        ("human", "Facts: {documents}\n\nResponse: {generation}")
    ])
    if (h_prompt | h_grader).invoke({"documents": "\n".join([d.page_content for d in state["documents"]]), "generation": state["generation"]}).binary_score.lower() == "no":
        return "not_grounded"

    a_grader = llm.with_structured_output(GradeAnswer)
    a_prompt = ChatPromptTemplate.from_messages([
        ("system", "Does this resolve the question? 'yes' or 'no'."),
        ("human", "Question: {question}\n\nAnswer: {generation}")
    ])
    if (a_prompt | a_grader).invoke({"question": state["question"], "generation": state["generation"]}).binary_score.lower() == "yes":
        return "useful"
    return "not_useful_web"

# --- COMPILE GRAPH ---
def get_rag_app():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("web_search", "generate")
    workflow.add_conditional_edges("grade_documents", route_after_grade, {"web_search": "web_search", "generate": "generate"})
    workflow.add_conditional_edges("generate", route_after_generate, {"useful": END, "not_grounded": "generate", "not_useful_web": "web_search"})
    return workflow.compile()

RAG_APP = get_rag_app()  # build the graph once, at startup, not per-request

# --- API ENDPOINTS ---
@app.post("/chat")
def chat(body: ChatIn):
    try:
        result = RAG_APP.invoke({"question": body.question})
        return {"answer": result.get("generation", "Error generating response."), "steps": result.get("steps", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "Agentic RAG Backend is Live!"}
