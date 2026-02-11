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
path2= 'sample_doc/Novartis_Report_card.pdf'
doc_intro = load_data(path2)

path1='sample_doc/Novartis_clinical_study.pdf'
docs=load_data(path1)
#print(docs
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vec_store = FAISS.from_documents(docs, embeddings)

# Save the vector store
vec_store.save_local("faiss_index")

vec_intro= FAISS.from_documents(doc_intro, embeddings)
vec_intro.save_local("faiss_index_report")

# COMMAND ----------

#loading arshad in vector DB
pdf_path='sample_doc/Fake_figure_2024-25.pdf'

docs_24=load_data(pdf_path)
vec_24_store = FAISS.from_documents(docs_24, embeddings)

# Save the vector store
vec_24_store.save_local("faiss_24_index")

# COMMAND ----------

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vec_store_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
vec_store_db2 = FAISS.load_local("faiss_24_index", embeddings, allow_dangerous_deserialization=True)
vec_store_db.merge_from(vec_store_db2)

vec_store_intro_db = FAISS.load_local("faiss_index_report", embeddings, allow_dangerous_deserialization=True)

vec_store_db.merge_from(vec_store_intro_db)



# COMMAND ----------

from langchain import hub
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

GROQ_API_KEY="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"
model= ChatGroq(model= "openai/gpt-oss-20b",groq_api_key=GROQ_API_KEY )
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(
       model, retrieval_qa_chat_prompt
   )

retrieval_chain = create_retrieval_chain(
        vec_store_db.as_retriever(),combine_docs_chain
   )


# COMMAND ----------

#asking from model
res2 = model.invoke('What is groq')
print(res2.content)


# COMMAND ----------

res = retrieval_chain.invoke({"input": "tell me the risk factors in short 5 points and what is the revenue details for 2024"})
print(res['answer'])

# COMMAND ----------

res = retrieval_chain.invoke({"input": "give me sales revenue details for 2023 & 2024 comparison"})
print(res['answer'])

# COMMAND ----------

