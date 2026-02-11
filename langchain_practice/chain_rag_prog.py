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

#pdf_path='vectordb/BigID_Intro1.pdf'
pdf_path='Arsad_Resume.pdf'
#pdf_path='Fake_bigID.pdf'

loader = PyPDFLoader(file_path=pdf_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
split_documents = text_splitter.split_documents(documents)


# COMMAND ----------


embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_documents, embeddings)

vectorstore.save_local("vectordb")
vectorstore.load_local("vectordb", embeddings, allow_dangerous_deserialization=True)

vector2= vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# COMMAND ----------

# 5. Retrieve relevant chunks for a query
query = "who is arsad"
retrieved_docs = vector2.get_relevant_documents(query)
print(f"Retrieved {len(retrieved_docs)} relevant document(s):")
for i, doc in enumerate(retrieved_docs):
    print(f"\nChunk {i+1} preview:\n{doc.page_content[:200]}...")



# COMMAND ----------

# 6. Set up the prompt and LLM for answer generation
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

GROQ_API_KEY="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"
prompt = PromptTemplate.from_template(
    "Given the context below, answer the user's question:\n\n{context}\n\nQuestion: {question}"
)
llm = ChatGroq(model= "openai/gpt-oss-20b",groq_api_key=GROQ_API_KEY )
output_parser = StrOutputParser()

# COMMAND ----------

#another task for websearch
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

urls=[
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
    "https://langchain-ai.github.io/langgraph/how-tos/map-reduce/"
]

web_docs=[WebBaseLoader(url).load() for url in urls]
web_docs
docs_list = [item for sublist in web_docs for item in sublist]
print(docs_list)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)

web_doc_splits = text_splitter.split_documents(docs_list)

## Add alll these text to vectordb
embeddings3=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

web_vec_store=FAISS.from_documents(
    documents=web_doc_splits,
    embedding=embeddings3
)

ret_web=web_vec_store.as_retriever()

# COMMAND ----------

web_response = ret_web.invoke("what is langgraph")

### Retriever To Retriever Tools
from langchain_core.tools.retriever import create_retriever_tool
#from langchain_community.agent_toolkits import create_retriever_tool

web_ret_tool=create_retriever_tool(
    web_response,
    "retriever_vector_db_log",
    "Search and run information about Langgraph"
)
web_ret_tool

tool=[vector2, web_ret_tool]


# COMMAND ----------

# 7. Prepare the context and run the RAG chain
context = "\n\n".join(doc.page_content for doc in retrieved_docs)

#context= "\n\n".join([resp for doc in response[0]])

chain_with_parser = prompt | llm |web_ret_tool | output_parser

#query = "who is arsad"
query= 'what is langgraph'
response = chain_with_parser.invoke({
    "context": context,
    "question": query
})

print("\nFinal Answer:")
print(response)

# COMMAND ----------

resp = chain_with_parser.invoke(
    {   'context' : context,
        'question':'what is the todays news in london'})
print(resp)