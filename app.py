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

# Create the temporary upload folder.
os.makedirs("uploads", exist_ok=True)

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)

app.config["UPLOAD_FOLDER"] = "uploads"

# Maximum size of one uploaded file.
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

CORS(app)


# Maximum number of documents in one session.
MAX_DOCUMENTS = 5


# Load Groq API key.
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    print("Warning: GROQ_API_KEY is not configured.")


# Initialize Groq.
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


# Initialize DuckDuckGo search.
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


# Store everything belonging to the current session.
class SessionState:

    def __init__(self):
        self.history = WindowedChatMessageHistory()

        # Current FAISS database.
        self.vectorstore = None

        # RAG function.
        self.rag_chain = None

        # List of uploaded filenames.
        self.documents = []

        # Store chunks separately for every document.
        #
        # Example:
        # {
        #     "resume.pdf": [chunk1, chunk2, ...],
        #     "paper.pdf": [chunk1, chunk2, ...]
        # }
        self.document_chunks = {}


store = {}


def get_session_state(session_id):

    if session_id not in store:
        store[session_id] = SessionState()

    return store[session_id]


# Decide whether a question needs web search.
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


# Normal chatbot mode.
def run_normal_chat(user_message, session_state):

    try:

        if llm is None:
            return "The AI model is currently unavailable."

        messages = [
            SystemMessage(
                content=(
                    "You are Eureka, a helpful and knowledgeable AI assistant.\n\n"
                    "Answer general questions using your own knowledge.\n\n"
                    "When web search results are provided, use them for current information.\n\n"
                    "Do not present information as current unless it is supported by the search results."
                )
            )
        ]

        messages.extend(
            session_state.history.messages
        )

        # Search only when the question appears time-sensitive.
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
                        "Answer the user's question using the search results above. "
                        "Do not make up information that is not supported by them."
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


# Create the RAG function.
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
                    "Answer the user's question using the provided document context.\n\n"
                    "Use the documents as the primary source and do not invent information.\n\n"
                    "The context may contain information from several different documents. "
                    "Combine information from them when necessary to answer the question.\n\n"
                    "If the answer cannot be found in the provided documents, clearly say "
                    "that the information is not available in the uploaded documents."
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

        try:

            documents = retriever.invoke(question)

            if not documents:

                return {
                    "answer": (
                        "I couldn't find relevant information "
                        "in the uploaded documents."
                    ),
                    "sources": []
                }

            context_parts = []

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

                    source_label = (
                        f"{filename} - Page {page + 1}"
                    )

                else:

                    source_label = filename

                context_parts.append(
                    f"Source: {source_label}\n"
                    f"{document.page_content}"
                )

            context = "\n\n---\n\n".join(
                context_parts
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

            # Collect unique sources.
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

                if source not in sources:
                    sources.append(source)

            return {
                "answer": str(answer),
                "sources": sources
            }

        except Exception as e:

            traceback.print_exc()

            return {
                "answer": f"Error while searching documents: {str(e)}",
                "sources": []
            }

    return answer_question


# Rebuild the FAISS database from all active documents.
def rebuild_vectorstore(session_state):

    if embeddings is None:
        raise RuntimeError(
            "Embedding model is not available."
        )

    all_chunks = []

    for filename in session_state.documents:

        chunks = session_state.document_chunks.get(
            filename,
            []
        )

        all_chunks.extend(chunks)

    # No documents remain.
    if not all_chunks:

        session_state.vectorstore = None
        session_state.rag_chain = None

        return

    print(
        f"Rebuilding FAISS index using "
        f"{len(all_chunks)} chunks..."
    )

    session_state.vectorstore = FAISS.from_documents(
        documents=all_chunks,
        embedding=embeddings
    )

    session_state.rag_chain = create_rag_chain(
        session_state.vectorstore
    )

    print("FAISS index rebuilt successfully.")


# Upload a new PDF or DOCX document.
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

    # Check maximum number of documents.
    if len(session_state.documents) >= MAX_DOCUMENTS:

        return jsonify(
            {
                "error": (
                    f"You can upload a maximum of "
                    f"{MAX_DOCUMENTS} documents."
                )
            }
        ), 400

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

    # Prevent duplicate filenames.
    if filename in session_state.documents:

        return jsonify(
            {
                "error": (
                    f"'{filename}' is already uploaded."
                )
            }
        ), 400

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    try:

        # Select the correct document loader.
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
            f"Loaded {len(documents)} pages from {filename}."
        )

        # Split the document into smaller chunks.
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

        # Store the filename in every chunk.
        for chunk in chunks:

            chunk.metadata["source"] = filename

        print(
            f"Created {len(chunks)} chunks from {filename}."
        )

        # Store chunks separately for this document.
        session_state.document_chunks[filename] = chunks

        # Add the filename to the active documents list.
        session_state.documents.append(
            filename
        )

        # Rebuild the FAISS database.
        rebuild_vectorstore(
            session_state
        )

        # Clear previous conversation because
        # the document collection has changed.
        session_state.history.clear()

        os.remove(filepath)

        return jsonify(
            {
                "success": True,
                "message": (
                    f"'{filename}' was added successfully."
                ),
                "documents": session_state.documents,
                "document_count": len(
                    session_state.documents
                )
            }
        ), 200

    except Exception as e:

        traceback.print_exc()

        # Remove temporary uploaded file.
        if os.path.exists(filepath):

            os.remove(filepath)

        # Remove partially stored document data.
        session_state.document_chunks.pop(
            filename,
            None
        )

        if filename in session_state.documents:

            session_state.documents.remove(
                filename
            )

        return jsonify(
            {
                "error": "Failed to process the document.",
                "details": str(e)
            }
        ), 500


# Return the currently uploaded documents.
@app.route("/documents", methods=["GET"])
def get_documents():

    session_state = get_session_state(
        "user_session_123"
    )

    return jsonify(
        {
            "documents": session_state.documents,
            "count": len(
                session_state.documents
            ),
            "max_documents": MAX_DOCUMENTS
        }
    ), 200


# Delete one specific document.
@app.route("/delete_document", methods=["POST"])
def delete_document():

    session_state = get_session_state(
        "user_session_123"
    )

    data = request.get_json(
        silent=True
    ) or {}

    filename = data.get(
        "filename"
    )

    if not filename:

        return jsonify(
            {
                "error": "No document name was provided."
            }
        ), 400

    filename = secure_filename(
        filename
    )

    # Check whether the document exists.
    if filename not in session_state.documents:

        return jsonify(
            {
                "error": (
                    f"'{filename}' is not currently uploaded."
                )
            }
        ), 404

    try:

        print(
            f"Deleting document: {filename}"
        )

        # Remove the document's chunks.
        session_state.document_chunks.pop(
            filename,
            None
        )

        # Remove the filename from the active list.
        session_state.documents.remove(
            filename
        )

        # Rebuild FAISS using only the remaining documents.
        rebuild_vectorstore(
            session_state
        )

        # Clear conversation history because
        # the available document context changed.
        session_state.history.clear()

        print(
            f"Document deleted: {filename}"
        )

        return jsonify(
            {
                "success": True,
                "message": (
                    f"'{filename}' was deleted successfully."
                ),
                "documents": session_state.documents,
                "document_count": len(
                    session_state.documents
                )
            }
        ), 200

    except Exception as e:

        traceback.print_exc()

        return jsonify(
            {
                "error": (
                    "Failed to delete the document."
                ),
                "details": str(e)
            }
        ), 500


# Keep this endpoint for compatibility.
# It can still remove everything if needed.
@app.route("/clear_document", methods=["POST"])
def clear_document():

    session_state = get_session_state(
        "user_session_123"
    )

    session_state.vectorstore = None
    session_state.rag_chain = None

    session_state.documents.clear()
    session_state.document_chunks.clear()

    session_state.history.clear()

    return jsonify(
        {
            "success": True,
            "message": "All document context cleared.",
            "documents": []
        }
    ), 200


# Chat endpoint.
@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(
        silent=True
    )

    if not data:

        return Response(
            "Error: Invalid JSON request.",
            status=400
        )

    user_message = data.get(
        "message"
    )

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

            # Use RAG when documents are available.
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

                # Add citations to the answer.
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