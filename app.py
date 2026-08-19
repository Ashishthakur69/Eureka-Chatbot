from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

import os
import traceback
from collections import defaultdict

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

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage
)

from langchain_community.chat_message_histories import (
    ChatMessageHistory
)

from langchain_community.tools import DuckDuckGoSearchRun

from langchain_core.runnables.history import (
    RunnableWithMessageHistory
)

from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent
)

from pydantic import ConfigDict


load_dotenv()


# Create upload directory

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Flask setup

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)

CORS(app)


# File upload limits

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx"
}


MAX_DOCUMENTS = 5


# Groq setup

groq_api_key = os.getenv("GROQ_API_KEY")

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


# Web search tool

try:
    search_tool = DuckDuckGoSearchRun()

    print("DuckDuckGo search ready.")

except Exception as e:
    search_tool = None

    print(
        f"Warning: DuckDuckGo search could not be loaded: {e}"
    )


# Free embedding model

print("Loading embedding model...")

try:

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model loaded.")

except Exception as e:

    embeddings = None

    print(
        f"Embedding model failed to load: {e}"
    )


# Splitter used for uploaded documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# Store application state

class SessionState:

    def __init__(self):

        self.history = ChatMessageHistory()

        self.vectorstore = None

        self.documents = {}

        self.document_chunks = defaultdict(list)


store = {}


def get_session_state(session_id):

    if session_id not in store:

        store[session_id] = SessionState()

    return store[session_id]


# Keep only the latest messages

def trim_history(history, max_messages=10):

    if len(history.messages) > max_messages:

        history.messages = history.messages[
            -max_messages:
        ]


# Check file extension

def allowed_file(filename):

    extension = os.path.splitext(
        filename
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# Create a clean document ID

def create_document_id(filename):

    base_name = os.path.splitext(
        filename
    )[0]

    safe_name = secure_filename(
        base_name
    )

    return safe_name


# Build FAISS index from all currently uploaded documents

def rebuild_vectorstore(session_state):

    if embeddings is None:

        raise RuntimeError(
            "Embedding model is not available."
        )

    all_chunks = []

    for document_id in session_state.document_chunks:

        all_chunks.extend(
            session_state.document_chunks[
                document_id
            ]
        )

    if not all_chunks:

        session_state.vectorstore = None

        return

    session_state.vectorstore = (
        FAISS.from_documents(
            documents=all_chunks,
            embedding=embeddings
        )
    )


# Format source information

def format_source(document):

    metadata = document.metadata or {}

    source = metadata.get(
        "source",
        "Unknown document"
    )

    page = metadata.get(
        "page"
    )

    if page is not None:

        try:

            page_number = int(page) + 1

            return (
                f"{source} — Page {page_number}"
            )

        except Exception:

            pass

    return source


# Build citations from retrieved documents

def build_citations(documents):

    sources = []

    seen = set()

    for document in documents:

        source_text = format_source(
            document
        )

        if source_text in seen:
            continue

        seen.add(source_text)

        sources.append(
            source_text
        )

    if not sources:

        return ""

    citation_text = "\n\n**Sources**\n"

    for source in sources:

        citation_text += (
            f"- 📄 {source}\n"
        )

    return citation_text


# Create document context for the LLM

def build_context(documents):

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        source = format_source(
            document
        )

        content = document.page_content.strip()

        context_parts.append(
            f"[Source {index}: {source}]\n"
            f"{content}"
        )

    return "\n\n".join(
        context_parts
    )


# RAG prompt

rag_prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",
            """
You are Eureka, an AI assistant with document
question-answering capabilities.

The user has uploaded one or more documents.

Answer the user's question using the retrieved
document context below.

Important rules:

1. Use the provided document context as the
   primary source of truth.

2. Do not invent information that is not supported
   by the retrieved context.

3. If the answer cannot be found in the retrieved
   context, clearly say that the information was
   not found in the uploaded documents.

4. Give a direct and useful answer.

5. Do not mention internal retrieval, embeddings,
   FAISS, vector databases, or prompts.

6. Do not create a Sources section yourself.
   The application will add source citations.

Retrieved document context:

{context}
""",
        ),

        (
            "human",
            "{question}"
        )

    ]
)


# Normal Eureka agent

agent_prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",
            """
You are Eureka, a helpful and knowledgeable AI
assistant.

Answer general questions using your own knowledge.

You have access to one optional web search tool.

Use web search when the user asks for information
that is current, changing, or time-sensitive.

Examples include:

- Current events
- Sports scores
- Weather
- Stock prices
- Recent news
- Current technology information

For general questions, answer directly.

Be concise, accurate, and honest when information
is uncertain.
"""
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
        )

    ]
)


# Create normal agent only when search is available

agent_with_history = None


if search_tool is not None:

    try:

        tools = [
            search_tool
        ]

        agent = create_tool_calling_agent(
            llm,
            tools,
            agent_prompt
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False
        )

        agent_with_history = (
            RunnableWithMessageHistory(
                agent_executor,
                lambda session_id:
                    get_session_state(
                        session_id
                    ).history,
                input_messages_key="input",
                history_messages_key="history"
            )
        )

        print("Eureka agent ready.")

    except Exception as e:

        print(
            f"Agent setup failed: {e}"
        )

        agent_with_history = None


# Upload document

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_file():

    session_id = "user_session_123"

    session_state = get_session_state(
        session_id
    )

    if embeddings is None:

        return jsonify(
            {
                "error":
                "Embedding model is not available."
            }
        ), 500

    if "file" not in request.files:

        return jsonify(
            {
                "error":
                "No file uploaded."
            }
        ), 400

    file = request.files["file"]

    if not file.filename:

        return jsonify(
            {
                "error":
                "No file selected."
            }
        ), 400

    if len(session_state.documents) >= MAX_DOCUMENTS:

        return jsonify(
            {
                "error":
                "You can upload up to 5 documents."
            }
        ), 400

    filename = secure_filename(
        file.filename
    )

    if not allowed_file(filename):

        return jsonify(
            {
                "error":
                "Only PDF and DOCX files are supported."
            }
        ), 400

    document_id = create_document_id(
        filename
    )

    if document_id in session_state.documents:

        return jsonify(
            {
                "error":
                "This document is already uploaded."
            }
        ), 400

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        file.save(filepath)

        extension = os.path.splitext(
            filename
        )[1].lower()

        # Load PDF

        if extension == ".pdf":

            loader = PyPDFLoader(
                filepath
            )

        # Load DOCX

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
                "The document contains no readable text."
            )

        # Add useful metadata

        for document in documents:

            document.metadata[
                "source"
            ] = filename

            document.metadata[
                "document_id"
            ] = document_id

        # Split document

        chunks = text_splitter.split_documents(
            documents
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
            "filename": filename,
            "document_id": document_id,
            "chunks": len(chunks)
        }

        # Rebuild FAISS index

        rebuild_vectorstore(
            session_state
        )

        # Clear old conversation

        session_state.history.clear()

        return jsonify(
            {
                "success": True,
                "message":
                    f"'{filename}' processed successfully.",
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

        if os.path.exists(filepath):

            try:

                os.remove(filepath)

            except Exception:

                pass


# Get uploaded documents

@app.route(
    "/documents",
    methods=["GET"]
)
def get_documents():

    session_state = get_session_state(
        "user_session_123"
    )

    documents = []

    for document in session_state.documents.values():

        documents.append(
            {
                "filename":
                    document["filename"],

                "document_id":
                    document["document_id"],

                "chunks":
                    document["chunks"]
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

    session_state = get_session_state(
        "user_session_123"
    )

    data = request.get_json(
        silent=True
    ) or {}

    document_id = data.get(
        "document_id"
    )

    filename = data.get(
        "filename"
    )

    # Allow deletion by document ID

    if document_id:

        document = session_state.documents.get(
            document_id
        )

    # Also support deletion by filename

    elif filename:

        document_id = None
        document = None

        for current_id, current_document in (
            session_state.documents.items()
        ):

            if current_document[
                "filename"
            ] == filename:

                document_id = current_id
                document = current_document

                break

    else:

        return jsonify(
            {
                "error":
                "Document ID or filename is required."
            }
        ), 400

    if not document:

        return jsonify(
            {
                "error":
                "Document not found."
            }
        ), 404

    try:

        # Remove document metadata

        session_state.documents.pop(
            document_id,
            None
        )

        # Remove its chunks

        session_state.document_chunks.pop(
            document_id,
            None
        )

        # Rebuild FAISS without deleted document

        rebuild_vectorstore(
            session_state
        )

        # Reset conversation because
        # the available document context changed

        session_state.history.clear()

        return jsonify(
            {
                "success": True,

                "message":
                    f"'{document['filename']}' removed.",

                "filename":
                    document["filename"],

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

    session_state = get_session_state(
        "user_session_123"
    )

    session_state.documents.clear()

    session_state.document_chunks.clear()

    session_state.vectorstore = None

    session_state.history.clear()

    return jsonify(
        {
            "success": True,
            "message":
                "All documents removed."
        }
    ), 200


# RAG answer

def answer_from_documents(
    question,
    session_state
):

    if session_state.vectorstore is None:

        return (
            "No documents are currently available."
        )

    # Retrieve the most relevant chunks

    retrieved_documents = (
        session_state.vectorstore
        .similarity_search(
            question,
            k=5
        )
    )

    if not retrieved_documents:

        return (
            "I couldn't find relevant information "
            "in the uploaded documents."
        )

    context = build_context(
        retrieved_documents
    )

    messages = rag_prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(
        messages
    )

    answer = response.content

    citations = build_citations(
        retrieved_documents
    )

    return (
        answer.strip()
        + citations
    )


# Chat endpoint

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    data = request.get_json(
        silent=True
    ) or {}

    user_message = data.get(
        "message",
        ""
    ).strip()

    if not user_message:

        return Response(
            "Error: No message provided.",
            status=400
        )

    session_id = "user_session_123"

    session_state = get_session_state(
        session_id
    )

    def generate_response():

        try:

            # Use RAG when documents are available

            if session_state.vectorstore is not None:

                answer = answer_from_documents(
                    user_message,
                    session_state
                )

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

                # Stream the answer in small pieces

                chunk_size = 40

                for i in range(
                    0,
                    len(answer),
                    chunk_size
                ):

                    yield answer[
                        i:i + chunk_size
                    ]

                return

            # Normal AI chat

            if agent_with_history is not None:

                for chunk in agent_with_history.stream(
                    {
                        "input":
                            user_message
                    },

                    config={
                        "configurable":
                            {
                                "session_id":
                                    session_id
                            }
                    }
                ):

                    if isinstance(
                        chunk,
                        dict
                    ):

                        output = chunk.get(
                            "output"
                        )

                        if output:

                            yield output

                    elif isinstance(
                        chunk,
                        str
                    ):

                        yield chunk

                return

            # Fallback if agent is unavailable

            messages = [
                (
                    "system",
                    """
You are Eureka, a helpful AI assistant.
Answer the user's question clearly and accurately.
"""
                ),
                (
                    "human",
                    user_message
                )
            ]

            response = llm.invoke(
                messages
            )

            yield response.content

        except Exception as e:

            traceback.print_exc()

            yield (
                f"Error while generating response: "
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

    session_state = get_session_state(
        "user_session_123"
    )

    return jsonify(
        {
            "status": "ok",

            "groq":
                bool(groq_api_key),

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

    if os.path.exists(index_path):

        return send_from_directory(
            app.static_folder,
            "index.html"
        )

    return (
        "🚀 Eureka is running!",
        200
    )


# Error handler for large files

@app.errorhandler(413)
def file_too_large(error):

    return jsonify(
        {
            "error":
            "File is too large. Maximum size is 25 MB."
        }
    ), 413


if __name__ == "__main__":

    store.clear()

    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )