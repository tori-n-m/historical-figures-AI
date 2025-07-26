from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

PDF_PATH = os.path.join(os.path.dirname(__file__), '../textbook/us-history-1492-to-2000.pdf')

def setup_rag_tool():
    # Load PDF and split into documents
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    # Create vector store from split documents
    vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return rag_chain

def answer_query(rag_chain, query):
    result = rag_chain({"query": query})
    return result["result"], result.get("source_documents", [])
