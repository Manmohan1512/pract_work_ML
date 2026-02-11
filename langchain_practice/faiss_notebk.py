# Databricks notebook source
pip install -r requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
#from langchain.embeddings import OpenAIEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader= TextLoader('speech.txt')
doc= loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(doc)
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print(type(docs))

# COMMAND ----------

embeddings3=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
doc_result= embeddings3.embed_documents(doc[0].page_content)

db=FAISS.from_documents(docs, embeddings3)
#print(doc_result)
db

# COMMAND ----------

# MAGIC %md
# MAGIC Query the faiss database

# COMMAND ----------

qry= 'what is collaborates with ML'
doc_re= db.similarity_search(qry)   
doc_score= db.similarity_search_with_score(qry)
print(doc_score)
print(doc_re)