from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from pypdf import PdfReader
import docx2txt

from pathlib import Path
from typing import List
from dotenv import load_dotenv
import os
import certifi


load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


def _get_writable_dir(subdir: str) -> Path:
    is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
    base = Path("/tmp") if is_serverless else Path(__file__).resolve().parent
    target = base / subdir
    try:
        target.mkdir(parents=True, exist_ok=True)
        return target
    except OSError:
        tmp_target = Path("/tmp") / subdir
        tmp_target.mkdir(parents=True, exist_ok=True)
        return tmp_target


UPLOADS_DIR = _get_writable_dir("uploads")
CHROMA_DIR = _get_writable_dir("chroma_db")


_vectorstore_instance = None


def get_vectorstore():
    global _vectorstore_instance
    if _vectorstore_instance is None:
        try:
            # Fix for older SQLite on Lambda/Vercel Linux
            try:
                __import__("pysqlite3")
                import sys
                sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
            except Exception:
                pass

            from langchain_chroma import Chroma
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=api_key
            )
            _vectorstore_instance = Chroma(
                collection_name="agentic_chatbot_docs",
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DIR)
            )
        except Exception as e:
            print(f"[RAG] Warning: Vector store failed to initialize: {e}")
            return None
    return _vectorstore_instance


class _LazyVectorstore:
    def __getattr__(self, name):
        vs = get_vectorstore()
        if vs is None:
            raise RuntimeError("Chroma vector store is not available in this environment.")
        return getattr(vs, name)


vectorstore = _LazyVectorstore()



def read_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"

        return text

    if suffix == ".docx":
        return docx2txt.process(file_path)

    if suffix in [".txt", ".md", ".py", ".csv"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV.")




def add_document_to_rag(file_path: str, thread_id: str):
    text = read_file_text(file_path)

    if not text.strip():
        raise ValueError("No text could be extracted from this file.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    docs: List[Document] = [
        Document(
            page_content=chunk,
            metadata={
                "thread_id": thread_id,
                "source": Path(file_path).name
            }
        )
        for chunk in chunks
    ]

    store = get_vectorstore()
    if store is None:
        raise ValueError("Chroma vector store is not available in this environment.")
    store.add_documents(docs)

    return {
        "filename": Path(file_path).name,
        "chunks": len(docs)
    }





def retrieve_from_rag(query: str, thread_id: str, k: int = 4) -> str:
    store = get_vectorstore()
    if store is None:
        return "No relevant uploaded document content found (vector store not initialized)."

    try:
        docs = store.similarity_search(
            query,
            k=k,
            filter={"thread_id": thread_id}
        )
    except Exception as e:
        return f"Error retrieving document content: {str(e)}"

    if not docs:
        return "No relevant uploaded document content found."

    results = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "uploaded document")
        results.append(
            f"[Source {i}: {source}]\n{doc.page_content}"
        )

    return "\n\n".join(results)