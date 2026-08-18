from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

import os
import traceback

from dotenv import load_dotenv
from werkzeug.utils import secure_filename


# ============================================================
# LangChain / AI Imports
# ============================================================

from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain_core.output_parsers import (
    StrOutputParser
)

from langchain_core.runnables import (
    RunnablePassthrough
)

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage
)

from langchain_community.chat_message_histories import (
    ChatMessageHistory
)

from langchain_community.tools import (
    DuckDuckGoSearchRun
)


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()


# ============================================================
# Flask Setup
# ============================================================

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
# Groq LLM
# ============================================================

groq_api_key = os.getenv("GROQ_API_KEY")


if not groq_api_key:

    print(
        "WARNING: GROQ_API_KEY is not configured."
    )


try:

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0.2
    )

    print(
        "--- Groq LLM initialized successfully ---"
    )

except Exception as e:

    print(
        f"--- ERROR initializing Groq LLM: {e} ---"
    )

    llm = None


# ============================================================
# DuckDuckGo Search
# ============================================================

try:

    search_tool = DuckDuckGoSearchRun()

    print(
        "--- DuckDuckGo search initialized successfully ---"
    )

except Exception as e:

    print(
        f"--- ERROR initializing DuckDuckGo: {e} ---"
    )

    search_tool = None


# ============================================================
# Hugging Face Embeddings
# ============================================================

print(
    "\n--- Initializing HuggingFace Embeddings Model ---"
)

try:

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print(
        "--- Embeddings Model Loaded Successfully ---\n"
    )

except Exception as e:

    print(
        f"--- ERROR: Failed to load embeddings model: {e} ---"
    )

    embeddings = None


# ============================================================
# Conversation History
# ============================================================

class WindowedChatMessageHistory(ChatMessageHistory):

    k: int = 5

    def add_message(
        self,
        message: BaseMessage
    ) -> None:

        super().add_message(message)

        if len(self.messages) > self.k * 2:

            self.messages = self.messages[
                -(self.k * 2):
            ]


# ============================================================
# Session State
# ============================================================

class SessionState:

    def __init__(self):

        self.history = WindowedChatMessageHistory()

        self.rag_chain = None


store = {}


def get_session_state(
    session_id: str
) -> SessionState:

    if session_id not in store:

        store[session_id] = SessionState()

    return store[session_id]


# ============================================================
# Determine Whether Web Search Is Needed
# ============================================================

def needs_web_search(
    user_message: str
) -> bool:

    search_keywords = [

        # Current information
        "latest",
        "recent",
        "today",
        "current",
        "now",
        "live",

        # News
        "news",
        "breaking",

        # Sports
        "score",
        "scores",
        "match",
        "matches",
        "live score",
        "result",
        "results",
        "standings",
        "schedule",

        # Finance
        "stock",
        "stocks",
        "share price",
        "stock price",
        "market price",

        # Weather
        "weather",
        "forecast",
        "temperature today",

        # Other changing information
        "price",
        "schedule",
        "release date",
        "released",
        "update",
        "updates",

        # Current year
        "2026"
    ]


    message_lower = user_message.lower()


    return any(
        keyword in message_lower
        for keyword in search_keywords
    )


# ============================================================
# Normal Chat
# ============================================================

def run_normal_chat(
    user_message: str,
    session_state: SessionState
):

    try:

        if llm is None:

            return (
                "The AI model is currently unavailable."
            )


        # ====================================================
        # System Message
        # ====================================================

        system_message = SystemMessage(
            content=(
                "You are Eureka, a helpful and "
                "knowledgeable AI assistant.\n\n"

                "Answer questions clearly and accurately.\n\n"

                "Use your own knowledge for general "
                "questions.\n\n"

                "When web search results are provided, "
                "use them as the source for current "
                "information.\n\n"

                "Do not claim information is live or "
                "current unless it is supported by "
                "the provided search results."
            )
        )


        # ====================================================
        # Conversation History
        # ====================================================

        messages = [
            system_message
        ]


        messages.extend(
            session_state.history.messages
        )


        # ====================================================
        # Decide Whether Search Is Needed
        # ====================================================

        search_required = needs_web_search(
            user_message
        )


        # ====================================================
        # WEB SEARCH MODE
        # ====================================================

        if (
            search_required
            and search_tool is not None
        ):

            print(
                "--- Searching DuckDuckGo ---"
            )


            try:

                search_results = (
                    search_tool.invoke(
                        user_message
                    )
                )


            except Exception as search_error:

                print(
                    f"--- Search error: {search_error} ---"
                )

                search_results = (
                    "Web search was unavailable."
                )


            # ------------------------------------------------
            # Send Search Results to Groq
            # ------------------------------------------------

            search_prompt = HumanMessage(
                content=(
                    f"User question:\n"
                    f"{user_message}\n\n"

                    f"DuckDuckGo search results:\n"
                    f"{search_results}\n\n"

                    "Answer the user's question using "
                    "the search results above.\n\n"

                    "Instructions:\n"
                    "- Give the most useful answer possible.\n"
                    "- For current information such as "
                    "sports scores, news, prices, weather, "
                    "or schedules, rely on the search results.\n"
                    "- Do not invent missing information.\n"
                    "- If the search results are insufficient, "
                    "say so honestly."
                )
            )


            messages.append(
                search_prompt
            )


            response = llm.invoke(
                messages
            )


        # ====================================================
        # NORMAL CHAT MODE
        # ====================================================

        else:

            print(
                "--- Using Groq without web search ---"
            )


            messages.append(
                HumanMessage(
                    content=user_message
                )
            )


            response = llm.invoke(
                messages
            )


        # ====================================================
        # Extract Response
        # ====================================================

        answer = response.content


        if isinstance(
            answer,
            list
        ):

            answer = "\n".join(
                str(item)
                for item in answer
            )


        answer = str(answer)


        # ====================================================
        # Save History
        # ====================================================

        session_state.history.add_user_message(
            user_message
        )

        session_state.history.add_ai_message(
            answer
        )


        return answer


    except Exception as e:

        traceback.print_exc()

        return (
            f"Error while generating response: {str(e)}"
        )


# ============================================================
# RAG Chain
# ============================================================

def create_rag_chain(
    vectorstore
):

    """
    Modern LCEL RAG pipeline:

    User Question
          ↓
    FAISS Retriever
          ↓
    Relevant Documents
          ↓
    Context
          ↓
    Groq
          ↓
    Answer
    """


    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )


    # ========================================================
    # RAG Prompt
    # ========================================================

    rag_prompt = ChatPromptTemplate.from_messages(
        [

            (
                "system",
                (
                    "You are Eureka, a helpful AI assistant.\n\n"

                    "Answer the user's question using "
                    "the provided document context.\n\n"

                    "Rules:\n"

                    "1. Use the uploaded document as "
                    "the primary source.\n"

                    "2. Do not invent information that "
                    "is not supported by the document.\n"

                    "3. If the answer cannot be found "
                    "in the document, clearly state that "
                    "the information is not available "
                    "in the uploaded document.\n"

                    "4. Give a clear and concise answer.\n\n"

                    "DOCUMENT CONTEXT:\n"
                    "{context}"
                )
            ),

            (
                "human",
                "{input}"
            )

        ]
    )


    # ========================================================
    # Format Documents
    # ========================================================

    def format_docs(
        docs
    ):

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )


    # ========================================================
    # LCEL RAG Chain
    # ========================================================

    rag_chain = (

        {
            "context": (
                retriever
                | format_docs
            ),

            "input": (
                RunnablePassthrough()
            )
        }

        | rag_prompt

        | llm

        | StrOutputParser()
    )


    return rag_chain


# ============================================================
# Upload Endpoint
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_file():

    session_state = get_session_state(
        "user_session_123"
    )


    # ========================================================
    # Check Embeddings
    # ========================================================

    if embeddings is None:

        return jsonify(
            {
                "error": (
                    "Embeddings model is not available."
                )
            }
        ), 500


    # ========================================================
    # Check File
    # ========================================================

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


    # ========================================================
    # Save File
    # ========================================================

    filename = secure_filename(
        file.filename
    )


    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    file.save(filepath)


    try:

        # ====================================================
        # Select Loader
        # ====================================================

        if filename.lower().endswith(
            ".pdf"
        ):

            loader = PyPDFLoader(
                filepath
            )


        elif filename.lower().endswith(
            ".docx"
        ):

            loader = Docx2txtLoader(
                filepath
            )


        else:

            os.remove(filepath)

            return jsonify(
                {
                    "error": (
                        "Unsupported file type. "
                        "Please upload PDF or DOCX."
                    )
                }
            ), 400


        # ====================================================
        # Load Documents
        # ====================================================

        docs = loader.load()


        if not docs:

            os.remove(filepath)

            return jsonify(
                {
                    "error": (
                        "Could not extract text "
                        "from the document."
                    )
                }
            ), 400


        print(
            f"--- Loaded {len(docs)} document pages ---"
        )


        # ====================================================
        # Split Documents
        # ====================================================

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
                    "error": (
                        "No text could be extracted "
                        "after splitting the document."
                    )
                }
            ), 400


        print(
            f"--- Created {len(splits)} document chunks ---"
        )


        # ====================================================
        # Create FAISS Vector Store
        # ====================================================

        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings
        )


        print(
            "--- FAISS vector store created successfully ---"
        )


        # ====================================================
        # Create RAG Chain
        # ====================================================

        session_state.rag_chain = (
            create_rag_chain(
                vectorstore
            )
        )


        # ====================================================
        # Clear Previous History
        # ====================================================

        session_state.history.clear()


        # ====================================================
        # Delete Temporary File
        # ====================================================

        if os.path.exists(filepath):

            os.remove(filepath)


        return jsonify(
            {
                "success": True,
                "message": (
                    f"File '{filename}' "
                    "processed successfully."
                )
            }
        ), 200


    except Exception as e:

        traceback.print_exc()


        if os.path.exists(filepath):

            os.remove(filepath)


        return jsonify(
            {
                "error": (
                    "Failed to process file."
                ),
                "details": str(e)
            }
        ), 500


# ============================================================
# Clear Document Endpoint
# ============================================================

@app.route(
    "/clear_document",
    methods=["POST"]
)
def clear_document():

    session_state = get_session_state(
        "user_session_123"
    )


    session_state.rag_chain = None

    session_state.history.clear()


    return jsonify(
        {
            "success": True,
            "message": (
                "Document context cleared."
            )
        }
    ), 200


# ============================================================
# Chat Endpoint
# ============================================================

@app.route(
    "/chat",
    methods=["POST"]
)
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


    # ========================================================
    # Generate Response
    # ========================================================

    def generate_response():

        try:

            # =================================================
            # RAG MODE
            # =================================================

            if session_state.rag_chain is not None:

                print(
                    "--- Using RAG document mode ---"
                )


                response_data = (
                    session_state.rag_chain.invoke(
                        user_message
                    )
                )


                if response_data:

                    yield str(
                        response_data
                    )

                else:

                    yield (
                        "No answer found in the document."
                    )


            # =================================================
            # NORMAL CHAT MODE
            # =================================================

            else:

                print(
                    "--- Using normal chat mode ---"
                )


                answer = run_normal_chat(
                    user_message,
                    session_state
                )


                yield answer


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
# Start Flask Server
# ============================================================

if __name__ == "__main__":

    store.clear()


    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )