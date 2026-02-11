# Databricks notebook source
pip install -r requirements.txt


# COMMAND ----------


dbutils.library.restartPython()


# COMMAND ----------

from langchain_groq import ChatGroq
GROQ_API_KEY="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"
#Gemma2-9b-it
model= ChatGroq(model= "openai/gpt-oss-20b",groq_api_key=GROQ_API_KEY )
model

# COMMAND ----------

from langchain_core.messages import HumanMessage, SystemMessage

msg= [
    SystemMessage(content="You are a helpful assistant that translates English to French"),
    HumanMessage(content="where is the capital of India")
                  
]

response= model.invoke(msg)


# COMMAND ----------

from langchain_core.output_parsers import StrOutputParser

parser= StrOutputParser()
parser.invoke(response)

# COMMAND ----------

#LCEL chain the component
chain= model|parser
chain.invoke(msg)

# COMMAND ----------

#prompt template
from langchain_core.prompts import ChatPromptTemplate

gen_template= "Translate following into {language}"
prompt=ChatPromptTemplate.from_messages(
    [("system",gen_template), ("user","{text}")]
)

# COMMAND ----------

result=prompt.invoke({"language":"French", "text":"Hello, How are your mom"})
result.to_messages()

# COMMAND ----------

chain= prompt|model|parser

chain.invoke({"language":"French", "text":"Hello, How are your mom"})