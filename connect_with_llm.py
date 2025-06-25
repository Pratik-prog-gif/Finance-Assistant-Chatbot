import os
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Validate token is loaded
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not loaded from .env")

# Load LLM
def load_llm(repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,  
        max_new_tokens=512                
    )
    return llm

# Prompt template
custom_prompt_template = """
Use the pieces of information provided in the context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Start the answer directly. No small talk, no preamble, no filler text.
"""

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

# Load FAISS DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_path = "vectorstore/db_faiss"
db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm("mistralai/Mistral-7B-Instruct-v0.3"),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt()}
)

# Ask question
user_question = input("What is your question? ")
response = qa_chain.invoke({'query': user_question})  

# Show response
print("\nAnswer:", response["result"])
print("\nSource Documents:\n")
for doc in response["source_documents"]:
    print(">", doc.page_content[:300])
 

