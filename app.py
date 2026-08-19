from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

import os
import traceback

from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

from langchain_community.chat_message_histories import (
    ChatMessageHistory
)

from langchain_community.tools import (
    DuckDuckGoSearchRun
)


load_dotenv()


# Flask setup

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)

CORS(app)


# Upload settings

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    25 * 1024 * 1024
)


# Supported files

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx"
}


MAX_DOCUMENTS = 5


# Groq API

groq_api_key = os.getenv(
    "GROQ_API_KEY"
)

if not groq_api_key:

    raise RuntimeError(
        "GROQ_API_KEY is not configured."
    )


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_api_key,
    temperature=0.2
)

print("Groq model loaded.")


# DuckDuckGo search

try:

    search_tool = DuckDuckGoSearchRun()

    print(
        "DuckDuckGo search ready."
    )

except Exception as e:

    search_tool = None

    print(
        f"DuckDuckGo search unavailable: {e}"
    )


# Free embedding model

print(
    "Loading embedding model..."
)

try:

    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    )

    print(
        "Embedding model loaded."
    )

except Exception as e:

    embeddings = None

    print(
        f"Embedding model failed: {e}"
    )


# Text splitting

text_splitter = (
    RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
)


# Session state

class SessionState:

    def __init__(self):

        self.history = (
            ChatMessageHistory()
        )

        self.vectorstore = None

        self.documents = {}

        self.document_chunks = {}


store = {}


def get_session_state(
    session_id
):

    if session_id not in store:

        store[session_id] = (
            SessionState()
        )

    return store[session_id]


# Keep conversation history small

def trim_history(
    history,
    max_messages=10
):

    if len(
        history.messages
    ) > max_messages:

        history.messages = (
            history.messages[
                -max_messages:
            ]
        )


# Check file type

def allowed_file(
    filename
):

    extension = os.path.splitext(
        filename
    )[1].lower()

    return (
        extension
        in ALLOWED_EXTENSIONS
    )


# Create document ID

def create_document_id(
    filename
):

    name = os.path.splitext(
        filename
    )[0]

    safe_name = secure_filename(
        name
    )

    return safe_name.lower()


# Rebuild FAISS index

def rebuild_vectorstore(
    session_state
):

    if embeddings is None:

        raise RuntimeError(
            "Embedding model is unavailable."
        )

    all_chunks = []

    for chunks in (
        session_state
        .document_chunks
        .values()
    ):

        all_chunks.extend(
            chunks
        )

    if not all_chunks:

        session_state.vectorstore = (
            None
        )

        return

    session_state.vectorstore = (
        FAISS.from_documents(
            documents=all_chunks,
            embedding=embeddings
        )
    )


# Format source citation

def format_source(
    document
):

    metadata = (
        document.metadata
        or {}
    )

    source = metadata.get(
        "source",
        "Unknown document"
    )

    page = metadata.get(
        "page"
    )

    if page is not None:

        try:

            page_number = (
                int(page) + 1
            )

            return (
                f"{source} — "
                f"Page {page_number}"
            )

        except Exception:

            pass

    return source


# Create citation section

def build_citations(
    documents
):

    citations = []

    seen = set()

    for document in documents:

        source = format_source(
            document
        )

        if source in seen:

            continue

        seen.add(source)

        citations.append(
            source
        )

    if not citations:

        return ""

    result = (
        "\n\n**Sources**\n"
    )

    for source in citations:

        result += (
            f"- 📄 {source}\n"
        )

    return result


# Build RAG context

def build_context(
    documents
):

    parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        source = format_source(
            document
        )

        content = (
            document
            .page_content
            .strip()
        )

        parts.append(
            f"[Source {index}: {source}]\n"
            f"{content}"
        )

    return "\n\n".join(
        parts
    )


# Decide whether web search is useful

def needs_web_search(
    question
):

    question_lower = (
        question
        .lower()
        .strip()
    )

    current_keywords = [

        "today",
        "current",
        "latest",
        "recent",
        "right now",
        "live",
        "breaking",
        "news",
        "score",
        "weather",
        "forecast",
        "stock price",
        "share price",
        "market today",
        "this week",
        "this month",
        "2026"

    ]

    return any(
        keyword
        in question_lower
        for keyword
        in current_keywords
    )


# RAG system prompt

RAG_SYSTEM_PROMPT = """
You are Eureka, a helpful AI assistant.

The user has uploaded documents and is asking a
question about them.

Use the retrieved document context as the primary
source of truth.

Rules:

1. Answer only from information supported by the
   retrieved document context.

2. Do not invent facts that are not present in the
   retrieved context.

3. If the answer cannot be found in the documents,
   clearly say that you could not find the answer
   in the uploaded documents.

4. Give a direct and useful answer.

5. Do not mention FAISS, embeddings, vector
   databases, retrieval pipelines, or prompts.

6. Do not create a Sources section yourself.
   The application will add citations automatically.

Retrieved document context:

{context}
"""


# Normal chat system prompt

NORMAL_SYSTEM_PROMPT = """
You are Eureka, a helpful and knowledgeable AI
assistant.

Answer questions clearly and naturally.

Use your own knowledge for general questions.

If web search results are provided, use them for
current or time-sensitive information.

Do not pretend information is current when it has
not been verified.

Be honest when information is uncertain.
"""


# Ask Groq about documents

def answer_from_documents(
    question,
    session_state
):

    if (
        session_state.vectorstore
        is None
    ):

        return (
            "No documents are currently "
            "available."
        )

    retrieved_documents = (
        session_state
        .vectorstore
        .similarity_search(
            question,
            k=5
        )
    )

    if not retrieved_documents:

        return (
            "I couldn't find relevant "
            "information in the uploaded "
            "documents."
        )

    context = build_context(
        retrieved_documents
    )

    system_message = (
        RAG_SYSTEM_PROMPT.format(
            context=context
        )
    )

    messages = [

        SystemMessage(
            content=system_message
        ),

        HumanMessage(
            content=question
        )

    ]

    response = llm.invoke(
        messages
    )

    answer = response.content

    if isinstance(
        answer,
        list
    ):

        answer = "\n".join(
            str(item)
            for item in answer
        )

    answer = str(
        answer
    ).strip()

    citations = (
        build_citations(
            retrieved_documents
        )
    )

    return (
        answer
        + citations
    )


# Normal Eureka chat

def answer_normal_question(
    question,
    session_state
):

    messages = [

        SystemMessage(
            content=NORMAL_SYSTEM_PROMPT
        )

    ]

    # Add recent conversation

    messages.extend(
        session_state
        .history
        .messages
    )

    search_results = None

    # Search only when the question
    # appears to require current information

    if (
        needs_web_search(question)
        and search_tool is not None
    ):

        try:

            print(
                f"Searching web for: {question}"
            )

            search_results = (
                search_tool.invoke(
                    question
                )
            )

        except Exception as e:

            print(
                f"Web search failed: {e}"
            )

            search_results = None

    # Add user question

    if search_results:

        user_content = (
            f"User question:\n"
            f"{question}\n\n"
            f"Web search results:\n"
            f"{search_results}\n\n"
            "Use the search results when "
            "they are relevant to the question."
        )

    else:

        user_content = question

    messages.append(
        HumanMessage(
            content=user_content
        )
    )

    response = llm.invoke(
        messages
    )

    answer = response.content

    if isinstance(
        answer,
        list
    ):

        answer = "\n".join(
            str(item)
            for item in answer
        )

    answer = str(
        answer
    ).strip()

    # Save conversation

    session_state.history.add_message(
        HumanMessage(
            content=question
        )
    )

    session_state.history.add_message(
        AIMessage(
            content=answer
        )
    )

    trim_history(
        session_state.history
    )

    return answer


# Upload endpoint

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_file():

    session_state = (
        get_session_state(
            "user_session_123"
        )
    )

    if embeddings is None:

        return jsonify(
            {
                "error":
                "Embedding model is unavailable."
            }
        ), 500

    if "file" not in request.files:

        return jsonify(
            {
                "error":
                "No file uploaded."
            }
        ), 400

    file = request.files[
        "file"
    ]

    if not file.filename:

        return jsonify(
            {
                "error":
                "No file selected."
            }
        ), 400

    if (
        len(
            session_state.documents
        )
        >= MAX_DOCUMENTS
    ):

        return jsonify(
            {
                "error":
                "You can upload up to 5 documents."
            }
        ), 400

    filename = secure_filename(
        file.filename
    )

    if not allowed_file(
        filename
    ):

        return jsonify(
            {
                "error":
                "Only PDF and DOCX files are supported."
            }
        ), 400

    document_id = (
        create_document_id(
            filename
        )
    )

    if (
        document_id
        in session_state.documents
    ):

        return jsonify(
            {
                "error":
                "This document is already uploaded."
            }
        ), 400

    filepath = os.path.join(
        app.config[
            "UPLOAD_FOLDER"
        ],
        filename
    )

    try:

        file.save(
            filepath
        )

        extension = (
            os.path.splitext(
                filename
            )[1]
            .lower()
        )

        if extension == ".pdf":

            loader = PyPDFLoader(
                filepath
            )

        elif extension == ".docx":

            loader = Docx2txtLoader(
                filepath
            )

        else:

            raise ValueError(
                "Unsupported document type."
            )

        documents = loader.load()

        if not documents:

            raise ValueError(
                "No readable text was found."
            )

        # Store document metadata

        for document in documents:

            document.metadata[
                "source"
            ] = filename

            document.metadata[
                "document_id"
            ] = document_id

        # Split into chunks

        chunks = (
            text_splitter
            .split_documents(
                documents
            )
        )

        if not chunks:

            raise ValueError(
                "Could not create document chunks."
            )

        # Store chunks

        session_state.document_chunks[
            document_id
        ] = chunks

        session_state.documents[
            document_id
        ] = {

            "filename":
                filename,

            "document_id":
                document_id,

            "chunks":
                len(chunks)

        }

        # Rebuild FAISS

        rebuild_vectorstore(
            session_state
        )

        # Reset previous conversation

        session_state.history.clear()

        return jsonify(
            {
                "success": True,

                "message":
                    f"'{filename}' "
                    "processed successfully.",

                "filename":
                    filename,

                "document_id":
                    document_id,

                "chunks":
                    len(chunks),

                "document_count":
                    len(
                        session_state.documents
                    ),

                "max_documents":
                    MAX_DOCUMENTS
            }
        ), 200

    except Exception as e:

        traceback.print_exc()

        session_state.documents.pop(
            document_id,
            None
        )

        session_state.document_chunks.pop(
            document_id,
            None
        )

        try:

            rebuild_vectorstore(
                session_state
            )

        except Exception:

            pass

        return jsonify(
            {
                "error":
                f"Failed to process document: {str(e)}"
            }
        ), 500

    finally:

        if os.path.exists(
            filepath
        ):

            try:

                os.remove(
                    filepath
                )

            except Exception:

                pass


# Get uploaded documents

@app.route(
    "/documents",
    methods=["GET"]
)
def get_documents():

    session_state = (
        get_session_state(
            "user_session_123"
        )
    )

    documents = []

    for document in (
        session_state
        .documents
        .values()
    ):

        documents.append(
            {
                "filename":
                    document[
                        "filename"
                    ],

                "document_id":
                    document[
                        "document_id"
                    ],

                "chunks":
                    document[
                        "chunks"
                    ]
            }
        )

    return jsonify(
        {
            "documents":
                documents,

            "count":
                len(documents),

            "max_documents":
                MAX_DOCUMENTS
        }
    )


# Delete one document

@app.route(
    "/delete_document",
    methods=["POST"]
)
def delete_document():

    session_state = (
        get_session_state(
            "user_session_123"
        )
    )

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    document_id = data.get(
        "document_id"
    )

    filename = data.get(
        "filename"
    )

    document = None

    # Find by document ID

    if document_id:

        document = (
            session_state
            .documents
            .get(
                document_id
            )
        )

    # Find by filename

    elif filename:

        for (
            current_id,
            current_document
        ) in (
            session_state
            .documents
            .items()
        ):

            if (
                current_document[
                    "filename"
                ]
                == filename
            ):

                document_id = (
                    current_id
                )

                document = (
                    current_document
                )

                break

    if not document:

        return jsonify(
            {
                "error":
                "Document not found."
            }
        ), 404

    try:

        filename = document[
            "filename"
        ]

        # Remove metadata

        session_state.documents.pop(
            document_id,
            None
        )

        # Remove chunks

        session_state.document_chunks.pop(
            document_id,
            None
        )

        # Rebuild FAISS

        rebuild_vectorstore(
            session_state
        )

        # Reset conversation

        session_state.history.clear()

        return jsonify(
            {
                "success": True,

                "message":
                    f"'{filename}' removed.",

                "filename":
                    filename,

                "document_count":
                    len(
                        session_state.documents
                    )
            }
        ), 200

    except Exception as e:

        traceback.print_exc()

        return jsonify(
            {
                "error":
                f"Failed to delete document: {str(e)}"
            }
        ), 500


# Clear all documents

@app.route(
    "/clear_document",
    methods=["POST"]
)
def clear_documents():

    session_state = (
        get_session_state(
            "user_session_123"
        )
    )

    session_state.documents.clear()

    session_state.document_chunks.clear()

    session_state.vectorstore = None

    session_state.history.clear()

    return jsonify(
        {
            "success": True,

            "message":
                "All documents removed.",

            "document_count":
                0
        }
    ), 200


# Chat endpoint

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    user_message = (
        data
        .get("message", "")
        .strip()
    )

    if not user_message:

        return Response(
            "Error: No message provided.",
            status=400
        )

    session_state = (
        get_session_state(
            "user_session_123"
        )
    )

    def generate_response():

        try:

            # Use document RAG when documents exist

            if (
                session_state.vectorstore
                is not None
            ):

                answer = (
                    answer_from_documents(
                        user_message,
                        session_state
                    )
                )

                # Save RAG conversation

                session_state.history.add_message(
                    HumanMessage(
                        content=user_message
                    )
                )

                session_state.history.add_message(
                    AIMessage(
                        content=answer
                    )
                )

                trim_history(
                    session_state.history
                )

            else:

                # Normal Eureka chat

                answer = (
                    answer_normal_question(
                        user_message,
                        session_state
                    )
                )

            # Send answer in chunks

            chunk_size = 40

            for start in range(
                0,
                len(answer),
                chunk_size
            ):

                yield answer[
                    start:
                    start + chunk_size
                ]

        except Exception as e:

            traceback.print_exc()

            yield (
                "Error while generating response: "
                f"{str(e)}"
            )

    return Response(
        generate_response(),
        mimetype="text/plain"
    )


# Health check

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    session_state = (
        get_session_state(
            "user_session_123"
        )
    )

    return jsonify(
        {
            "status":
                "ok",

            "groq":
                bool(
                    groq_api_key
                ),

            "embeddings":
                embeddings is not None,

            "documents":
                len(
                    session_state.documents
                ),

            "max_documents":
                MAX_DOCUMENTS
        }
    )


# Main page

@app.route("/")
def serve_frontend():

    index_path = os.path.join(
        app.static_folder,
        "index.html"
    )

    if os.path.exists(
        index_path
    ):

        return send_from_directory(
            app.static_folder,
            "index.html"
        )

    return (
        "🚀 Eureka is running!",
        200
    )


# Handle files larger than 25 MB

@app.errorhandler(413)
def file_too_large(
    error
):

    return jsonify(
        {
            "error":
            "File is too large. "
            "Maximum size is 25 MB."
        }
    ), 413


# Start application

if __name__ == "__main__":

    store.clear()

    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )