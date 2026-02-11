# Databricks notebook source
pip install -r requirements.txt


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

#text loader
from langchain_community.document_loaders import TextLoader
ld= TextLoader('speech.txt')
txt = ld.load()
print(txt)

# COMMAND ----------

#reading pdf file

from langchain_community.document_loaders import PyPDFLoader
pdf_ld=PyPDFLoader('Arsad_Resume.pdf')
pdf=pdf_ld.load()
print(pdf)

# COMMAND ----------

#web-based loader
from langchain_community.document_loaders import WebBaseLoader
import bs4
web_ld= WebBaseLoader( web_path='https://docs.databricks.com/aws/en/getting-started/free-edition',
                      bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                          class_=("post_title", "post_content", "post_meta")
                      )))

web = web_ld.load()

print(web)

# COMMAND ----------

#wikipedia loader
from langchain_community.document_loaders import WikipediaLoader
wiki_ld=WikipediaLoader(query='Future of AI',load_max_docs=1).load()

print(wiki_ld)


# COMMAND ----------

# MAGIC %md
# MAGIC Text recursively split

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter
txt_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
txt_split=txt_splitter.split_documents(pdf)
print(type(txt_split))
print(txt_split[0])
print(txt_split[1])
#pdf_split=txt_splitter.split_documents(pdf)

# COMMAND ----------

from langchain_text_splitters import HTMLHeaderTextSplitter
url="https://docs.databricks.com/aws/en/getting-started/free-edition"
hd_split_on=[
("h1","header"),
("h2","subheader")
]
html_splitter= HTMLHeaderTextSplitter(hd_split_on)
html_ld= html_splitter.split_text(url)
#split web url data


html_ld

# COMMAND ----------

# MAGIC %md
# MAGIC Embedding Techniques
# MAGIC

# COMMAND ----------

import os
from dotenv import load_dotenv
load_dotenv() #load all environment variables
OPENAI_API_KEY="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"

print(OPENAI_API_KEY)

#os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")


# COMMAND ----------

from langchain_openai import OpenAIEmbeddings
embeddings=OpenAIEmbeddings(model='text-embedding-3-small')
txt='this is the random text for embedding practice'
embeddings.embed_query(txt)

# COMMAND ----------

# MAGIC %md
# MAGIC Ollama embedding model

# COMMAND ----------

from langchain_community.embeddings import OllamaEmbeddings

embeddings2=(
    OllamaEmbeddings(
        model="gemma3:4b",
       # temperature=0.01
))
embeddings2

# COMMAND ----------

r1 = embeddings2.embed_documents(
    ['alpha is starting character of alphabet','beta is starting character of alphabet']
)
r1
#result_ret = r1.embed_query('alpha is starting character of alphabet')
#print(result_ret)

# COMMAND ----------

# MAGIC %md
# MAGIC embedding with huggingface

# COMMAND ----------


from langchain_huggingface import HuggingFaceEmbeddings
embeddings3=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# COMMAND ----------

txt="this is first text parameters and this is also random sentence"

query_result=embeddings3.embed_query(txt)
len(query_result)

# COMMAND ----------

print(txt[0])
print(type(txt[0]))

# COMMAND ----------

#print(txt[0].page_content)
doc_result= embeddings3.embed_documents(txt[0].page_content)

print(doc_result)