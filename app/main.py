from fastapi import FastAPI
import pandas as pd
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse
from pathlib import Path
from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

app = FastAPI(title="Financial Analysis Service")

VECTOR_SIZE = 1536
COLLECTION_NAME = "fin-docs"
DATA_FOLDER = "./data"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

qdrant_client = QdrantClient(location=":memory:")
qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("human",
     "Eres un asistente para tareas de pregunta-respuesta. Usa los siguientes fragmentos "
     "de contexto recuperados para responder a la pregunta. Si no sabes la respuesta, di "
     "que no lo sabes. Responde en como máximo tres frases y de forma concisa.\n\n"
     "Pregunta: {question}\n"
     "Contexto: {context}\n"
     "Respuesta:"),
])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/technical-analysis")
def technical_analysis(symbol: str = None):
    prices = pd.read_csv("./notebooks/data/prices.csv")
    available_symbols = prices["Symbol"].unique().tolist()
    if symbol is None:
        return {"message": f"Please choose a symbol from the available symbols: {available_symbols}"}

    result = generate_plot(symbol)
    image = Path(result["url"])
    return FileResponse(path=image, media_type="image/png", filename=f"{symbol}_macd.png")


def generate_plot(symbol: str):
    prices = pd.read_csv("./notebooks/data/prices.csv")
    df = prices[prices["Symbol"] == symbol].copy()
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    df["Close"].plot(ax=ax, color='#374151', alpha=0.2, secondary_y=True)
    macd.plot(ax=ax, color='blue')
    signal.plot(ax=ax, color='red')

    ax.set_title(f"MACD and Signal for {symbol}")
    ax.set_xlabel("Date")
    ax.set_ylabel("MACD")
    ax.right_ax.set_ylabel("Close Price")

    Path("./outputs").mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    plt.savefig(f"./outputs/{symbol}macd{timestamp}.png")
    plt.close(fig)

    return {
        "message": f"MACD and Signal plot for {symbol} saved as ./outputs/{symbol}macd{timestamp}.png",
        "url": f"./outputs/{symbol}macd{timestamp}.png"
    }


@app.post("/index_data")
async def index_data():
    pages = []
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DATA_FOLDER, filename)
            loader = PyPDFLoader(file_path)
            async for page in loader.alazy_load():
                pages.append(page)

    if not pages:
        return {"message": "No PDF files found in data folder."}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(pages)

    document_ids = vector_store.add_documents(all_splits)

    return {
        "message": "Data indexed successfully.",
        "pages_loaded": len(pages),
        "chunks_indexed": len(all_splits),
        "sample_ids": document_ids[:3],
    }


@app.post("/query")
async def query_rag(query: str):
    retrieved_docs = vector_store.similarity_search_with_score(query, k=5)

    if not retrieved_docs:
        return {"answer": "No se encontraron documentos relevantes. Asegúrate de indexar datos primero con /index_data."}

    docs_content = "\n\n".join(doc.page_content for doc, _score in retrieved_docs)

    messages = prompt.invoke({"question": query, "context": docs_content})
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": [
            {"content": doc.page_content[:200], "score": score}
            for doc, score in retrieved_docs
        ],
    }
