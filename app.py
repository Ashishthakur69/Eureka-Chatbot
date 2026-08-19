from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

import os
import traceback
from typing import ClassVar
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun


load_dotenv()

# Create the folder used for temporary uploaded files.
os.makedirs("uploads", exist_ok=True)

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)

app.config["UPLOAD_FOLDER"] = "uploads"

CORS(app)


# Load the Groq API key from the environment.
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    print("Warning: GROQ_API_KEY is not configured.")


try:
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0.2
    )

    print("Groq model loaded.")

except Exception as e:
    print(f"Could not initialize Groq: {e}")
    llm = None


# Initialize DuckDuckGo for questions that need current information.
try:
    search_tool = DuckDuckGoSearchRun()
    print("DuckDuckGo search ready.")

except Exception as e:
    print(f"Could not initialize DuckDuckGo: {e}")
    search_tool = None


# Load the free Hugging Face embedding model.
print("Loading embedding model...")

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("Embedding model loaded.")

except Exception as e:
    print(f"Could not load embedding model: {e}")
    embeddings = None


# Keep only the most recent five conversations.
class WindowedChatMessageHistory(ChatMessageHistory):

    k: ClassVar[int] = 5

    def add_message(self, message: BaseMessage) -> None:
        super().add_message(message)

        if len(self.messages) > self.k * 2:
            self.messages = self.messages[-(self.k * 2):]


# Store the current conversation and document context.
class SessionState:

    def __init__(self):
        self.history = WindowedChatMessageHistory()
        self.rag_chain = None


store = {}


def get_session_state(session_id):
    if session_id not in store:
        store[session_id] = SessionState()

    return store[session_id]


# Decide whether a question is likely to need a web search.
def needs_web_search(user_message):

    search_keywords = [
        "latest",
        "recent",
        "today",
        "current",
        "now",
        "live",
        "news",
        "breaking",
        "score",
        "scores",
        "match",
        "matches",
        "result",
        "results",
        "standings",
        "schedule",
        "stock",
        "stocks",
        "share price",
        "stock price",
        "market price",
        "weather",
        "forecast",
        "temperature today",
        "price",
        "release date",
        "released",
        "update",
        "updates",
        "2026"
    ]

    message = user_message.lower()

    return any(
        keyword in message
        for keyword in search_keywords
    )


# Handle normal conversations when no document is uploaded.
def run_normal_chat(user_message, session_state):

    try:
        if llm is None:
            return "The AI model is currently unavailable."

        messages = [
            SystemMessage(
                content=(
                    "You are Eureka, a helpful and "
                    "knowledgeable AI assistant.\n\n"
                    "Answer general questions using your "
                    "own knowledge.\n\n"
                    "When web search results are provided, "
                    "use them for current information.\n\n"
                    "Do not present information as current "
                    "unless it is supported by the search results."
                )
            )
        ]

        messages.extend(
            session_state.history.messages
        )

        # Search the web only when the question appears time-sensitive.
        if (
            needs_web_search(user_message)
            and search_tool is not None
        ):

            print("Searching DuckDuckGo...")

            try:
                search_results = search_tool.invoke(
                    user_message
                )

            except Exception as e:
                print(f"Search failed: {e}")
                search_results = "Web search was unavailable."

            messages.append(
                HumanMessage(
                    content=(
                        f"User question:\n"
                        f"{user_message}\n\n"
                        f"Search results:\n"
                        f"{search_results}\n\n"
                        "Answer the user's question using "
                        "the search results above. Do not "
                        "make up information that is not "
                        "supported by them."
                    )
                )
            )

        else:

            messages.append(
                HumanMessage(
                    content=user_message
                )
            )

        response = llm.invoke(messages)

        answer = response.content

        if isinstance(answer, list):
            answer = "\n".join(
                str(item)
                for item in answer
            )

        answer = str(answer)

        session_state.history.add_user_message(
            user_message
        )

        session_state.history.add_ai_message(
            answer
        )

        return answer

    except Exception as e:
        traceback.print_exc()
        return f"Error while generating response: {str(e)}"


# Create the RAG function used after a document is uploaded.
def create_rag_chain(vectorstore):

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are Eureka, a helpful AI assistant.\n\n"
                    "Answer the user's question using the "
                    "provided document context.\n\n"
                    "Use the document as the primary source "
                    "and do not invent information.\n\n"
                    "If the answer cannot be found in the "
                    "document, clearly say that the information "
                    "is not available in the uploaded document."
                )
            ),
            (
                "human",
                (
                    "Document context:\n"
                    "{context}\n\n"
                    "Question:\n"
                    "{question}"
                )
            )
        ]
    )

    def answer_question(question):

        # Retrieve the most relevant document chunks.
        documents = retriever.invoke(question)

        if not documents:
            return {
                "answer": (
                    "I couldn't find relevant information "
                    "in the uploaded document."
                ),
                "sources": []
            }

        # Combine the retrieved chunks into the context for the LLM.
        context = "\n\n---\n\n".join(
            document.page_content
            for document in documents
        )

        messages = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        response = llm.invoke(messages)

        answer = response.content

        if isinstance(answer, list):
            answer = "\n".join(
                str(item)
                for item in answer
            )

        # Get the source file and page from the document metadata.
        sources = []

        for document in documents:

            source_path = document.metadata.get(
                "source",
                "Unknown document"
            )

            filename = os.path.basename(
                source_path
            )

            page = document.metadata.get(
                "page"
            )

            if page is not None:
                page_number = page + 1
                label = (
                    f"{filename} - Page {page_number}"
                )
            else:
                page_number = None
                label = filename

            source = {
                "file": filename,
                "page": page_number,
                "label": label
            }

            # Don't show the same source more than once.
            if source not in sources:
                sources.append(source)

        return {
            "answer": str(answer),
            "sources": sources
        }

    return answer_question


# Upload a PDF or DOCX file and create its vector index.
@app.route("/upload", methods=["POST"])
def upload_file():

    session_state = get_session_state(
        "user_session_123"
    )

    if embeddings is None:
        return jsonify(
            {
                "error": "Embedding model is not available."
            }
        ), 500

    if "file" not in request.files:
        return jsonify(
            {
                "error": "No file was uploaded."
            }
        ), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify(
            {
                "error": "No file was selected."
            }
        ), 400

    filename = secure_filename(
        file.filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    try:

        # Choose the loader based on the uploaded file type.
        if filename.lower().endswith(".pdf"):

            loader = PyPDFLoader(filepath)

        elif filename.lower().endswith(".docx"):

            loader = Docx2txtLoader(filepath)

        else:

            os.remove(filepath)

            return jsonify(
                {
                    "error": (
                        "Unsupported file type. "
                        "Please upload a PDF or DOCX file."
                    )
                }
            ), 400

        documents = loader.load()

        if not documents:

            os.remove(filepath)

            return jsonify(
                {
                    "error": (
                        "No readable text was found "
                        "in the document."
                    )
                }
            ), 400

        print(
            f"Loaded {len(documents)} document pages."
        )

        # Split large documents into smaller chunks for retrieval.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(
            documents
        )

        if not chunks:

            os.remove(filepath)

            return jsonify(
                {
                    "error": (
                        "The document could not be "
                        "split into text chunks."
                    )
                }
            ), 400

        print(
            f"Created {len(chunks)} text chunks."
        )

        # Create the FAISS vector database from the chunks.
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        print("FAISS vector store created.")

        # Create a RAG function for the uploaded document.
        session_state.rag_chain = create_rag_chain(
            vectorstore
        )

        # Start a fresh conversation for the new document.
        session_state.history.clear()

        # The document is no longer needed after indexing.
        os.remove(filepath)

        return jsonify(
            {
                "success": True,
                "message": (
                    f"'{filename}' was processed successfully."
                )
            }
        ), 200

    except Exception as e:

        traceback.print_exc()

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify(
            {
                "error": "Failed to process the document.",
                "details": str(e)
            }
        ), 500


# Remove the current document from the session.
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
            "message": "Document context cleared."
        }
    ), 200


# Handle chat messages.
@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    data = request.json

    if not data:
        return Response(
            "Error: Invalid JSON request.",
            status=400
        )

    user_message = data.get("message")

    if not user_message:
        return Response(
            "Error: No message provided.",
            status=400
        )

    session_state = get_session_state(
        "user_session_123"
    )

    def generate_response():

        try:

            # Use the uploaded document when RAG is active.
            if session_state.rag_chain is not None:

                print("Using document RAG.")

                result = session_state.rag_chain(
                    user_message
                )

                answer = result.get(
                    "answer",
                    "No answer found."
                )

                sources = result.get(
                    "sources",
                    []
                )

                # Add the retrieved document sources to the answer.
                if sources:

                    answer += "\n\nSources:\n"

                    for source in sources:

                        answer += (
                            f"• {source['label']}\n"
                        )

                yield answer

            else:

                print("Using normal chat.")

                answer = run_normal_chat(
                    user_message,
                    session_state
                )

                yield answer

        except Exception as e:

            traceback.print_exc()

            yield f"Error: {str(e)}"

    return Response(
        generate_response(),
        mimetype="text/event-stream"
    )


# Serve the frontend.
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


if __name__ == "__main__":

    store.clear()

    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )