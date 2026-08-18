from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import traceback
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# --- LangChain & AI imports ---
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun

from langchain.agents import AgentExecutor, create_tool_calling_agent


# ============================================================
# Setup
# ============================================================

load_dotenv()

if not os.path.exists("uploads"):
    os.makedirs("uploads")

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)

app.config["UPLOAD_FOLDER"] = "uploads"

CORS(app)


# ============================================================
# Load API Key & Setup LLM
# ============================================================

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    print("WARNING: GROQ_API_KEY is not set.")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_api_key
)


# ============================================================
# Pre-load Embedding Model
# ============================================================

print("\n--- Initializing HuggingFace Embeddings Model ---")

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("--- Embeddings Model Loaded Successfully ---\n")

except Exception as e:
    print(
        f"--- ERROR: Failed to load embeddings model: {e} ---"
    )

    embeddings = None


# ============================================================
# State Management
# ============================================================

class WindowedChatMessageHistory(ChatMessageHistory):

    k: int = 5

    def add_message(self, message: BaseMessage) -> None:

        super().add_message(message)

        if len(self.messages) > self.k * 2:
            self.messages = self.messages[-(self.k * 2):]


class SessionState:

    def __init__(self):

        self.history = WindowedChatMessageHistory()

        self.rag_chain = None


store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:

    if session_id not in store:

        store[session_id] = SessionState()

    return store[session_id].history


def get_session_state(session_id: str) -> SessionState:

    if session_id not in store:

        store[session_id] = SessionState()

    return store[session_id]


# ============================================================
# Agent Prompt & Setup
# ============================================================

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are Eureka, a helpful and knowledgeable assistant.\n"

                "Answer user questions using your own knowledge first "
                "whenever possible.\n\n"

                "You have access to ONE tool: 'duckduckgo_search'.\n"

                "Use this tool ONLY when the query clearly requires "
                "up-to-date or real-time information.\n"

                "Examples:\n"
                "- Current events\n"
                "- Weather forecasts\n"
                "- Sports scores\n"
                "- Stock prices\n\n"

                "If the question is general knowledge, answer directly "
                "without searching.\n"

                "Always synthesize results and be honest if you cannot "
                "find an answer."
            ),
        ),

        MessagesPlaceholder(
            variable_name="history"
        ),

        (
            "human",
            "{input}"
        ),

        MessagesPlaceholder(
            variable_name="agent_scratchpad"
        ),
    ]
)


# ============================================================
# DuckDuckGo Search Agent
# ============================================================

tools = [
    DuckDuckGoSearchRun()
]


agent = create_tool_calling_agent(
    llm,
    tools,
    agent_prompt
)


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# ============================================================
# RAG Chain
# ============================================================

def create_rag_chain(vectorstore):

    """
    Creates a modern LCEL-based Retrieval-Augmented Generation chain.

    Flow:

    User Question
        ↓
    FAISS Retriever
        ↓
    Relevant Documents
        ↓
    Context + Question
        ↓
    Groq LLM
        ↓
    Answer
    """

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )


    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are Eureka, a helpful AI assistant.\n\n"

                    "Answer the user's question using the provided "
                    "document context.\n\n"

                    "Rules:\n"
                    "1. Use the document context as the primary source.\n"
                    "2. Do not invent information that is not supported "
                    "by the document.\n"
                    "3. If the answer cannot be found in the document, "
                    "clearly say that the information is not available "
                    "in the uploaded document.\n"
                    "4. Give a clear and concise answer.\n\n"

                    "DOCUMENT CONTEXT:\n"
                    "{context}"
                ),
            ),

            (
                "human",
                "{input}"
            ),
        ]
    )


    def format_docs(docs):

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )


    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
        }

        | rag_prompt

        | llm

        | StrOutputParser()
    )


    return rag_chain


# ============================================================
# Upload Endpoint
# ============================================================

@app.route("/upload", methods=["POST"])
def upload_file():

    session_state = get_session_state(
        "user_session_123"
    )


    # Check embeddings

    if embeddings is None:

        return jsonify(
            {
                "error": "Embeddings model is not available."
            }
        ), 500


    # Check file

    if "file" not in request.files:

        return jsonify(
            {
                "error": "No file part"
            }
        ), 400


    file = request.files["file"]


    if file.filename == "":

        return jsonify(
            {
                "error": "No selected file"
            }
        ), 400


    if file:

        filename = secure_filename(
            file.filename
        )


        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )


        file.save(filepath)


        try:

            # ------------------------------------------------
            # Load document
            # ------------------------------------------------

            if filename.lower().endswith(".pdf"):

                loader = PyPDFLoader(
                    filepath
                )

            elif filename.lower().endswith(".docx"):

                loader = Docx2txtLoader(
                    filepath
                )

            else:

                os.remove(filepath)

                return jsonify(
                    {
                        "error": "Unsupported file type. "
                                 "Please upload PDF or DOCX."
                    }
                ), 400


            # ------------------------------------------------
            # Extract documents
            # ------------------------------------------------

            docs = loader.load()


            if not docs:

                os.remove(filepath)

                return jsonify(
                    {
                        "error": "Could not extract text "
                                 "from the document."
                    }
                ), 400


            # ------------------------------------------------
            # Split text
            # ------------------------------------------------

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )


            splits = text_splitter.split_documents(
                docs
            )


            if not splits:

                os.remove(filepath)

                return jsonify(
                    {
                        "error": "No text could be extracted "
                                 "after splitting the document."
                    }
                ), 400


            print(
                f"--- Created {len(splits)} document chunks ---"
            )


            # ------------------------------------------------
            # Create FAISS Vector Store
            # ------------------------------------------------

            vectorstore = FAISS.from_documents(
                documents=splits,
                embedding=embeddings
            )


            print(
                "--- FAISS vector store created successfully ---"
            )


            # ------------------------------------------------
            # Create RAG Chain
            # ------------------------------------------------

            session_state.rag_chain = create_rag_chain(
                vectorstore
            )


            # ------------------------------------------------
            # Clear previous conversation history
            # ------------------------------------------------

            session_state.history.clear()


            # ------------------------------------------------
            # Remove uploaded file
            # ------------------------------------------------

            if os.path.exists(filepath):

                os.remove(filepath)


            return jsonify(
                {
                    "success": True,
                    "message": (
                        f"File '{filename}' processed "
                        "successfully."
                    )
                }
            ), 200


        except Exception as e:

            traceback.print_exc()


            if os.path.exists(filepath):

                os.remove(filepath)


            return jsonify(
                {
                    "error": "Failed to process file.",
                    "details": str(e)
                }
            ), 500


# ============================================================
# Clear Document Endpoint
# ============================================================

@app.route("/clear_document", methods=["POST"])
def clear_document():

    session_state = get_session_state(
        "user_session_123"
    )


    session_state.rag_chain = None

    session_state.history.clear()


    return jsonify(
        {
            "success": True,
            "message": "Document context cleared."
        }
    ), 200


# ============================================================
# Chat Endpoint
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.json


    if not data:

        return Response(
            "Error: Invalid JSON request",
            status=400
        )


    user_message = data.get(
        "message"
    )


    if not user_message:

        return Response(
            "Error: No message provided",
            status=400
        )


    session_state = get_session_state(
        "user_session_123"
    )


    def generate_response():

        try:

            # =================================================
            # DOCUMENT / RAG MODE
            # =================================================

            if session_state.rag_chain:

                print(
                    "--- Using RAG document mode ---"
                )


                response_data = (
                    session_state.rag_chain.invoke(
                        user_message
                    )
                )


                if response_data:

                    yield response_data

                else:

                    yield (
                        "No answer found in the document."
                    )


            # =================================================
            # NORMAL CHAT / AGENT MODE
            # =================================================

            else:

                print(
                    "--- Using normal agent mode ---"
                )


                for chunk in agent_with_history.stream(

                    {
                        "input": user_message
                    },

                    config={
                        "configurable": {
                            "session_id":
                                "user_session_123"
                        }
                    },
                ):

                    if "output" in chunk:

                        yield chunk["output"]


        except Exception as e:

            traceback.print_exc()

            yield (
                f"Error: {str(e)}"
            )


    return Response(
        generate_response(),
        mimetype="text/event-stream"
    )


# ============================================================
# Root Route
# ============================================================

@app.route("/")
def serve_frontend():

    index_path = os.path.join(
        app.static_folder,
        "index.html"
    )


    if os.path.exists(index_path):

        return send_from_directory(
            app.static_folder,
            "index.html"
        )


    return (
        "🚀 Eureka Chatbot is running!",
        200
    )


# ============================================================
# Server Run
# ============================================================

if __name__ == "__main__":

    store.clear()


    app.run(
        host="0.0.0.0",
        port=7860,
        debug=True
    )