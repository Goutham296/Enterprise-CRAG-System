import os
import warnings
import logging
import requests
import threading

# Suppress harmless warnings (LangChain, Torchvision, Flask-Limiter)
warnings.filterwarnings("ignore")

# Enterprise Structured Logging & Observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgenticRAG")

from typing import TypedDict, List
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# Load environment variables (.env file)
load_dotenv()

# Enable LLM Caching (Cost Optimization & Latency Reduction)
set_llm_cache(InMemoryCache())

# ============================================================================
# 1. HTML FRONTEND (Served by Flask)
# ============================================================================
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Agentic QA Assistant</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f9fafb; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }
        input, button { padding: 10px; margin: 5px 0; border: 1px solid #ccc; border-radius: 4px; }
        button { background: #0070f3; color: white; cursor: pointer; border: none; }
        button:hover { background: #0051a8; }
        .response-box { background: #f3f4f6; padding: 15px; border-left: 4px solid #0070f3; margin-top: 15px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Enterprise Document AI</h1>
    
    <div class="card">
        <h3>1. Ingest Document</h3>
        <input type="file" id="fileInput" accept=".txt,.pdf">
        <input type="text" id="webhookInput" style="width: 80%;" placeholder="(Optional) Webhook URL for async notification...">
        <button onclick="uploadDocument()">Upload & Vectorize</button>
        <p id="uploadStatus"></p>
    </div>

    <div class="card">
        <h3>2. Agentic Document Q&A</h3>
        <input type="text" id="queryInput" style="width: 80%;" placeholder="Ask a question about the documents...">
        <button onclick="askQuestion()">Ask Agent</button>
        
        <div id="responseContainer" style="display: none;">
            <h4>Answer:</h4>
            <div id="answerText" class="response-box"></div>
            <h4>Executive Summary:</h4>
            <div id="summaryText" class="response-box" style="border-left-color: #10b981;"></div>
        </div>
    </div>

    <script>
        // Fetch current host dynamically
        const API_URL = window.location.origin + "/api/v1";
        const HEADERS = { "Authorization": "Bearer super-secure-api-key-123" }; 

        async function uploadDocument() {
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files[0]) return alert("Select a file first!");
            
            document.getElementById('uploadStatus').innerText = "Processing semantic chunks...";
            
            const formData = new FormData();
            formData.append("file", fileInput.files[0]);
            
            const webhookUrl = document.getElementById('webhookInput').value;
            if(webhookUrl) formData.append("webhook_url", webhookUrl);

            const response = await fetch(`${API_URL}/ingest`, {
                method: "POST", headers: HEADERS, body: formData
            });
            const data = await response.json();
            document.getElementById('uploadStatus').innerText = data.message || data.error;
        }

        async function askQuestion() {
            const query = document.getElementById('queryInput').value;
            if (!query) return;

            document.getElementById('responseContainer').style.display = "block";
            document.getElementById('answerText').innerText = "Agent workflow executing (Retrieving -> Grading -> Generating)...";
            document.getElementById('summaryText').innerText = "...";

            const response = await fetch(`${API_URL}/ask`, {
                method: "POST",
                headers: { ...HEADERS, "Content-Type": "application/json" },
                body: JSON.stringify({ question: query })
            });
            
            const data = await response.json();
            document.getElementById('answerText').innerText = data.answer || data.error;
            document.getElementById('summaryText').innerText = data.summary || "";
        }
    </script>
</body>
</html>
"""

# ============================================================================
# 2. VECTOR DATABASE & INGESTION (RAG PIPELINE)
# ============================================================================
class EnterpriseRAGPipeline:
    def __init__(self, pg_uri: str):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name="enterprise_knowledge",
            connection=pg_uri,
            use_jsonb=True,
        )

    def ingest_document(self, file_path: str, filename: str):
        """Ingests a document, applies semantic chunking, and stores in pgVector."""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file format")
            
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        for split in splits:
            split.metadata['source_file'] = filename
            split.metadata['doc_type'] = 'business_document'

        self.vector_store.add_documents(splits)
        return len(splits)

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": 4})

# ============================================================================
# 3. LANGGRAPH AGENTIC WORKFLOW (Self-Correcting CRAG)
# ============================================================================
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    summary: str
    retries: int

class AgenticWorkflow:
    def __init__(self, rag_pipeline: EnterpriseRAGPipeline):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.retriever = rag_pipeline.get_retriever()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("grade_documents", self.grade_documents_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("summarize", self.summarize_node)
        workflow.add_node("rewrite_query", self.rewrite_query_node)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents", self.decide_to_generate,
            {"generate": "generate", "rewrite_query": "rewrite_query"}
        )
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("generate", "summarize")
        workflow.add_edge("summarize", END)

        return workflow.compile()

    def retrieve_node(self, state: GraphState):
        docs = self.retriever.invoke(state["question"])
        return {"documents": docs, "retries": state.get("retries", 0)}

    def grade_documents_node(self, state: GraphState):
        class Grade(BaseModel):
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        structured_llm_grader = self.llm.with_structured_output(Grade)
        prompt = ChatPromptTemplate.from_template(
            "You are a grader assessing relevance of a retrieved document to a user question.\n"
            "Document: {document}\nQuestion: {question}\n"
            "If the document contains keyword(s) or semantic meaning related to the question, grade it as 'yes'."
        )
        grader = prompt | structured_llm_grader

        filtered_docs = []
        for doc in state["documents"]:
            score = grader.invoke({"question": state["question"], "document": doc.page_content})
            if score.binary_score == "yes":
                filtered_docs.append(doc)
                
        return {"documents": filtered_docs}

    def decide_to_generate(self, state: GraphState):
        if not state["documents"] and state["retries"] < 2:
            return "rewrite_query"
        return "generate"

    def rewrite_query_node(self, state: GraphState):
        prompt = ChatPromptTemplate.from_template(
            "You are an expert at optimizing search queries for vector databases.\n"
            "Look at the initial question and formulate an improved version.\n"
            "Initial question: {question}\nImproved question:"
        )
        rewriter = prompt | self.llm
        improved_question = rewriter.invoke({"question": state["question"]})
        return {"question": improved_question.content, "retries": state["retries"] + 1}

    def generate_node(self, state: GraphState):
        context = "\n\n".join([doc.page_content for doc in state["documents"]])
        prompt = ChatPromptTemplate.from_template(
            "Use the following pieces of retrieved context to answer the question.\n"
            "If you don't know the answer, say that you don't know.\n"
            "Question: {question}\nContext: {context}\nAnswer:"
        )
        chain = prompt | self.llm
        response = chain.invoke({"question": state["question"], "context": context})
        return {"generation": response.content}

    def summarize_node(self, state: GraphState):
        prompt = ChatPromptTemplate.from_template("Provide a one-sentence executive summary of the following text:\n{text}")
        chain = prompt | self.llm
        summary = chain.invoke({"text": state["generation"]})
        return {"summary": summary.content}
        
    def run(self, question: str):
        logger.info(f"Starting Agentic Workflow for query: '{question}'")
        result = self.graph.invoke({"question": question, "documents": [], "retries": 0})
        logger.info(f"Workflow complete. Docs used: {len(result.get('documents', []))}. Retries: {result.get('retries')}")
        return {"answer": result.get("generation"), "summary": result.get("summary"), "docs_used": len(result.get("documents", []))}

# ============================================================================
# 4. FLASK SECURE API
# ============================================================================
app = Flask(__name__)
CORS(app)
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])
app.config['UPLOAD_FOLDER'], app.config['MAX_CONTENT_LENGTH'] = 'uploads', 16 * 1024 * 1024
API_KEY, ALLOWED_EXTENSIONS = os.getenv("API_SECRET_KEY", "super-secure-api-key-123"), {'txt', 'pdf'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

pg_uri = os.getenv("PG_VECTOR_URI", "postgresql+psycopg://postgres:password@localhost:5433/enterprise_rag")
rag_pipeline = EnterpriseRAGPipeline(pg_uri=pg_uri)
agent_workflow = AgenticWorkflow(rag_pipeline=rag_pipeline)

def require_api_key(func):
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header != f"Bearer {API_KEY}": return jsonify({"error": "Unauthorized"}), 401
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

@app.after_request
def add_security_headers(response):
    """OWASP API Security Best Practices: Strict HTTP Headers"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

def fire_webhook(url, payload):
    """Helper function to fire webhooks asynchronously."""
    try:
        requests.post(url, json=payload, timeout=3)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Webhook delivery failed: {e}")

@app.route('/', methods=['GET'])
def index(): return render_template_string(INDEX_HTML)

@app.route('/api/v1/ingest', methods=['POST'])
@require_api_key
@limiter.limit("10 per minute")
def upload_file():
    file = request.files.get('file')
    if not file or not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
        return jsonify({"error": "Invalid file"}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)
    try:
        chunks = rag_pipeline.ingest_document(filepath, file.filename)
        logger.info(f"Document {file.filename} ingested successfully. {chunks} chunks embedded.")
        
        # Event-Driven Architecture: Asynchronous Webhook Trigger
        webhook_url = request.form.get("webhook_url")
        if webhook_url:
            payload = {"event": "ingestion_complete", "filename": file.filename, "chunks": chunks}
            # Fire and forget in a background thread so the API responds instantly
            threading.Thread(target=fire_webhook, args=(webhook_url, payload)).start()
            
        return jsonify({"message": f"Successfully ingested {file.filename}", "chunks_embedded": chunks}), 201
    except Exception as e: return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath): os.remove(filepath)

@app.route('/api/v1/ask', methods=['POST'])
@require_api_key
@limiter.limit("20 per minute")
def ask_question():
    data = request.get_json()
    if not data or 'question' not in data: return jsonify({"error": "Missing 'question'"}), 400
    try: return jsonify(agent_workflow.run(data['question'])), 200
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Dynamically bind to the port assigned by the Cloud provider
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)