from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import traceback

# --- All necessary imports for the final hybrid system ---
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Imports for the new, stable agent ---
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_tool_calling_agent

# -------------------------
# Setup
# -------------------------
load_dotenv()

if not os.path.exists('uploads'):
    os.makedirs('uploads')

app = Flask(__name__, static_folder="../frontend", static_url_path="")
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)

# Load API Key & Setup LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

# Pre-load the Embedding Model
print("\n--- Initializing HuggingFace Embeddings Model ---")
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("--- Embeddings Model Loaded Successfully ---\n")
except Exception as e:
    print(f"--- ERROR: Failed to load embeddings model: {e} ---")
    embeddings = None

# --- State Management ---
# ✨ FIX: We'll now store both the chat history and the RAG chain in a session object.
class SessionState:
    def __init__(self):
        self.history = WindowedChatMessageHistory()
        self.rag_chain = None

store = {}

class WindowedChatMessageHistory(ChatMessageHistory):
    k: int = 5
    def add_message(self, message: BaseMessage) -> None:
        super().add_message(message)
        if len(self.messages) > self.k * 2:
            self.messages = self.messages[-(self.k * 2):]

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = SessionState()
    return store[session_id].history

def get_session_state(session_id: str) -> SessionState:
    if session_id not in store:
        store[session_id] = SessionState()
    return store[session_id]


# --- ** THE FINAL, IMPROVED AGENT PROMPT ** ---
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", (
       "You are Eureka, a helpful and knowledgeable assistant.\n"
        "Answer user questions using your own knowledge first whenever possible.\n\n"
        "You have access to ONE tool: 'duckduckgo_search'.\n"
        "Use this tool ONLY when the query clearly requires up-to-date or real-time information.\n"
        "Examples of when to use search:\n"
        "- Current events or news\n"
        "- Live sports scores, match schedules\n"
        "- Weather forecasts\n"
        "- Stock market updates\n"
        "- Other recent or time-sensitive information\n\n"
        "If the question is general knowledge (e.g., history, science, definitions, coding, math), "
        "answer directly without searching.\n\n"
        "When using the search tool, make your queries as specific as possible.\n"
        "Always evaluate results critically and synthesize them before responding.\n"
        "If you cannot find a reliable answer, say so honestly."
    )),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# --- Stable Agent with Live Search ---
tools = [DuckDuckGoSearchRun()]
agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# -------------------------
# UPLOAD ENDPOINT
# -------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    # ✨ FIX: We get the session state instead of using a global variable.
    session_state = get_session_state("user_session_123")

    if embeddings is None: return jsonify({"error": "Embeddings model is not available."}), 500
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            if filename.endswith('.pdf'): loader = PyPDFLoader(filepath)
            elif filename.endswith('.docx'): loader = Docx2txtLoader(filepath)
            else:
                os.remove(filepath)
                return jsonify({"error": "Unsupported file type"}), 400
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            
            # ✨ FIX: The RAG chain is now stored in the user's session state.
            session_state.rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
            session_state.history.clear() # Clear history when a new doc is uploaded.
            
            os.remove(filepath)
            return jsonify({"success": True, "message": f"File '{filename}' processed successfully."}), 200
        except Exception as e:
            traceback.print_exc()
            if os.path.exists(filepath): os.remove(filepath)
            return jsonify({"error": f"Failed to process file. See server logs."}), 500

# -------------------------
# CLEAR DOCUMENT ENDPOINT
# -------------------------
@app.route("/clear_document", methods=["POST"])
def clear_document():
    # ✨ FIX: We get the session state and clear its specific properties.
    session_state = get_session_state("user_session_123")
    session_state.rag_chain = None
    session_state.history.clear()
    return jsonify({"success": True, "message": "Document context and chat history cleared."}), 200

# -------------------------
# The Hybrid Chat Logic
# -------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")
    if not user_message:
        return Response("Error: No message provided", status=400)

    # ✨ FIX: Get the current session to check for a RAG chain.
    session_state = get_session_state("user_session_123")

    def generate_response():
        try:
            # ✨ FIX: Check for the rag_chain within the session, not globally.
            if session_state.rag_chain:
                response_data = session_state.rag_chain.invoke(user_message)
                yield response_data.get('result', "Could not find an answer in the document.")
            else:
                for chunk in agent_with_history.stream(
                    {"input": user_message},
                    config={"configurable": {"session_id": "user_session_123"}}
                ):
                    if "output" in chunk:
                        yield chunk["output"]
        except Exception as e:
            traceback.print_exc()
            yield f"Error: {str(e)}"

    return Response(generate_response(), mimetype='text/event-stream')

# -------------------------
# Frontend Route & Server Run
# -------------------------
@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)