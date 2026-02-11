# Databricks notebook source
pip install -r requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings

loader= TextLoader('speech.txt')
doc= loader.load()
#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#docs = text_splitter.split_documents(doc)
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
doc

# COMMAND ----------

#split
from langchain_text_splitters import RecursiveCharacterTextSplitter
txt_splitter= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
split_doc= txt_splitter.split_documents(doc)
split_doc

# COMMAND ----------

#embedding
from langchain_huggingface import HuggingFaceEmbeddings

embeddings3=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#doc_result= embeddings3.embed_documents(doc[0].page_content)
vectordb= Chroma.from_documents(split_doc, embeddings3)
vectordb

# COMMAND ----------

#query the vector db
qry= 'what is collaboration of ML'
docs= vectordb.similarity_search(qry)
docs[0].page_content

# COMMAND ----------

#saving vector db locally
vectordb = Chroma.from_documents(split_doc, embeddings3, persist_directory="./vectordb")
#vectordb.persist()

# COMMAND ----------

#load from local disk
db2= Chroma(persist_directory="./vectordb", embedding_function=embeddings3)
docs= db2.similarity_search_with_score(qry)
docs[0][0].page_content

# COMMAND ----------


docs= db2.similarity_search(qry)
docs

# COMMAND ----------

#retriver options
retriver= vectordb.as_retriever()
retriver.invoke(qry)[0].page_content