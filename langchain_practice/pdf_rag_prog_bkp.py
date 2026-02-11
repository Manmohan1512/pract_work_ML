# Databricks notebook source
pip install -r requirements.txt


# COMMAND ----------

dbutils.library.restartPython()


# COMMAND ----------

#text loader to extract data from source--> dividing into chunks
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
#from langchain.embeddings import OpenAIEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

def load_data(file_path):
    loader = PyPDFLoader(file_path)
    doc = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    ret_docs = text_splitter.split_documents(doc)
    return ret_docs


# COMMAND ----------

#Loading the BigID PDF
path1= 'vectordb/BigID_Intro1.pdf'
docs=load_data(path1)
#print(docs
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vec_store = FAISS.from_documents(docs, embeddings)

# Save the vector store
vec_store.save_local("faiss_index")

# Load the vector store
#nw_vec_store = FAISS.load_local(
#       "faiss_index", embeddings, allow_dangerous_deserialization=True
#   )


# COMMAND ----------

#loading arshad in vector DB
pdf_path='Arsad_Resume.pdf'
print(pdf_path)
docs_cv=load_data(pdf_path)
#print('This is what is print',docs)
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
cv_vec_store = FAISS.from_documents(docs_cv, embeddings)

# Save the vector store
cv_vec_store.save_local("faiss_cv_index")

# Load the vector store
#nw_cv_vec_store = FAISS.load_local(
#       "faiss_cv_index", embeddings, allow_dangerous_deserialization=True
#   )

# COMMAND ----------

vectorstore1 = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
vectorstore2 = FAISS.load_local("faiss_cv_index", embeddings, allow_dangerous_deserialization=True)
vectorstore1.merge_from(vectorstore2)
retriever = vectorstore1#.as_retriever()

# COMMAND ----------

from langchain import hub
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


GROQ_API_KEY="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"
#Gemma2-9b-it
model= ChatGroq(model= "openai/gpt-oss-20b",groq_api_key=GROQ_API_KEY )

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(
       model, retrieval_qa_chat_prompt
   )

retrieval_chain = create_retrieval_chain(
        retriever.as_retriever(),combine_docs_chain
   )



# COMMAND ----------

#res2 = combine_docs_chain.invoke({"context": docs, "messages": [
#            HumanMessage(content="what are the three main points to solve the housing crisis?")
#        ]})
res2 = model.invoke('what is Big ID')
print(res2.content)

# COMMAND ----------

res = retrieval_chain.invoke({"input": "BigID summary"})
print(res['answer'])

# COMMAND ----------

res = retrieval_chain.invoke({"input": "who is Arsad"})
print(res['answer'])

# COMMAND ----------

