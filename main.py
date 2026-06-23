from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

llm = None
vectorstore = None

persist_directory = Path(__file__).parent / "resources/vectorstore"


def database():
    global llm, vectorstore

    # Initialize Embedding + Vector DB
    if vectorstore is None:
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma(
            collection_name="real_estate",
            embedding_function=embedding,
            persist_directory=str(persist_directory)
        )

    # Initialize LLM
    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )


def processurl(urls):
    global vectorstore

    print("Initializing database...")
    database()

    print("Loading documents from URLs...")

    try:
        loader = WebBaseLoader(
            urls,
            requests_kwargs={
                "headers": {
                    "User-Agent": "Mozilla/5.0"
                }
            }
        )

        docs = loader.load()

    except Exception as e:
        raise Exception(f"Error loading URLs: {e}")

    print("Documents loaded:", len(docs))

    if not docs:
        raise ValueError("No documents loaded from URLs.")

    print("Splitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    print("Chunks created:", len(chunks))

    if not chunks:
        raise ValueError("No chunks created from documents.")

    print("Clearing old vector database...")

    # Prevent duplicate embeddings
    print("Clearing old vector database...")

    if vectorstore._collection.count() > 0:
        ids = vectorstore.get()["ids"]

        if ids:
            vectorstore.delete(ids=ids)

    print("Adding documents to vectorstore...")

    uuids = [str(uuid4()) for _ in range(len(chunks))]

    vectorstore.add_documents(
        documents=chunks,
        ids=uuids
    )

    print("Documents successfully added to vectorstore.")


def generate(query):
    global vectorstore

    # Ensure database + llm initialized
    database()

    if vectorstore is None:
        raise Exception("Vectorstore not initialized. Please process URLs first.")

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found.", []

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    sources = list({
        doc.metadata.get("source", "Unknown")
        for doc in docs
    })

    prompt = ChatPromptTemplate.from_template(
        """
        

        Answer the question only from the provided context.

        If the answer is not present in context, say:
        "I could not find relevant information."

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": query
    })

    return answer, sources
