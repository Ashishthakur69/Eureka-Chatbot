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
    HumanMessage,
    ToolMessage
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

        # Keep last 5 user/assistant exchanges
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
# Normal Chat LLM With Tools
# ============================================================

if llm is not None:

    try:

        llm_with_tools = llm.bind_tools(
            [search_tool]
        )

        print(
            "--- Groq tool calling initialized ---"
        )

    except Exception as e:

        print(
            f"--- WARNING: Tool calling unavailable: {e} ---"
        )

        llm_with_tools = llm

else:

    llm_with_tools = None


# ============================================================
# Normal Chat Function
# ============================================================

def run_normal_chat(
    user_message: str,
    session_state: SessionState
):

    try:

        if llm_with_tools is None:

            return (
                "The AI model is currently unavailable."
            )


        # ----------------------------------------------------
        # System instruction
        # ----------------------------------------------------

        system_message = SystemMessage(
            content=(
                "You are Eureka, a helpful and "
                "knowledgeable AI assistant.\n\n"

                "Answer general questions using your "
                "own knowledge.\n\n"

                "You have access to a DuckDuckGo search "
                "tool. Use it when the user asks for "
                "current, recent, real-time, or changing "
                "information.\n\n"

                "Examples that may require search:\n"
                "- Current events\n"
                "- Latest news\n"
                "- Weather\n"
                "- Sports scores\n"
                "- Stock prices\n"
                "- Current technology information\n"
                "- Recent releases or updates\n\n"

                "For normal general-knowledge questions, "
                "answer directly without searching.\n\n"

                "When using search results, synthesize "
                "the information and do not blindly copy it."
            )
        )


        # ----------------------------------------------------
        # Previous conversation
        # ----------------------------------------------------

        messages = [
            system_message
        ]

        messages.extend(
            session_state.history.messages
        )

        messages.append(
            HumanMessage(
                content=user_message
            )
        )


        # ----------------------------------------------------
        # First LLM call
        # ----------------------------------------------------

        response = llm_with_tools.invoke(
            messages
        )


        # ----------------------------------------------------
        # Check whether the model requested a tool
        # ----------------------------------------------------

        if hasattr(response, "tool_calls") and response.tool_calls:

            tool_messages = [
                response
            ]


            # ------------------------------------------------
            # Execute requested tools
            # ------------------------------------------------

            for tool_call in response.tool_calls:

                tool_name = tool_call["name"]

                tool_args = tool_call.get(
                    "args",
                    {}
                )


                if (
                    tool_name == "duckduckgo_search"
                    and search_tool is not None
                ):

                    try:

                        search_result = (
                            search_tool.invoke(
                                tool_args
                            )
                        )

                    except Exception:

                        # Some versions expect a string
                        query = tool_args.get(
                            "query",
                            ""
                        )

                        search_result = (
                            search_tool.invoke(
                                query
                            )
                        )


                    tool_messages.append(
                        ToolMessage(
                            content=str(
                                search_result
                            ),
                            tool_call_id=tool_call["id"]
                        )
                    )


            # ------------------------------------------------
            # Second LLM call with search results
            # ------------------------------------------------

            final_messages = (
                messages + tool_messages
            )


            final_response = llm.invoke(
                final_messages
            )


            answer = final_response.content


        else:

            answer = response.content


        # ----------------------------------------------------
        # Save conversation
        # ----------------------------------------------------

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
    Creates a modern LCEL RAG chain.

    User question
        ↓
    FAISS Retriever
        ↓
    Relevant document chunks
        ↓
    Prompt
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


    # --------------------------------------------------------
    # RAG Prompt
    # --------------------------------------------------------

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are Eureka, a helpful AI assistant.\n\n"

                    "Answer the user's question using the "
                    "provided document context.\n\n"

                    "Important rules:\n"

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


    # --------------------------------------------------------
    # Format Retrieved Documents
    # --------------------------------------------------------

    def format_docs(
        docs
    ):

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )


    # --------------------------------------------------------
    # LCEL RAG Chain
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Check Embeddings
    # --------------------------------------------------------

    if embeddings is None:

        return jsonify(
            {
                "error": (
                    "Embeddings model is not available."
                )
            }
        ), 500


    # --------------------------------------------------------
    # Check File
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Save File
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )


    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    file.save(filepath)


    try:

        # ----------------------------------------------------
        # Select Document Loader
        # ----------------------------------------------------

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
                        "Please upload a PDF or DOCX file."
                    )
                }
            ), 400


        # ----------------------------------------------------
        # Load Document
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Split Document
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Create FAISS Vector Store
        # ----------------------------------------------------

        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings
        )


        print(
            "--- FAISS vector store created successfully ---"
        )


        # ----------------------------------------------------
        # Create RAG Chain
        # ----------------------------------------------------

        session_state.rag_chain = (
            create_rag_chain(
                vectorstore
            )
        )


        # ----------------------------------------------------
        # Clear Previous Chat History
        # ----------------------------------------------------

        session_state.history.clear()


        # ----------------------------------------------------
        # Remove Temporary File
        # ----------------------------------------------------

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


    def generate_response():

        try:

            # =================================================
            # DOCUMENT / RAG MODE
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
# Run Server
# ============================================================

if __name__ == "__main__":

    store.clear()


    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )