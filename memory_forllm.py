from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Load PDF files from a directory
dir_path="."
def load_pdf_from_directory(path):
    loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_from_directory(dir_path)
print('length of documents:', len(documents))

#Create a text splitter
def create_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(data)
    return chunks

chunks = create_chunks(documents)
print('length of chunks:', len(chunks))

#Create embeddings
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
    
embedding_model=get_embeddings()

#store embeddings in faiss
db_path = "vectorstore/db_faiss"
os.makedirs(db_path, exist_ok=True)
db= FAISS.from_documents(chunks, embedding_model)
db.save_local(db_path)


