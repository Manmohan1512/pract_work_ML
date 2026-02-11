# Databricks notebook source
pip install -r requirements.txt

# COMMAND ----------

dbutils.library.restartPython()


# COMMAND ----------

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
GROQ_API_KEY="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"

model= ChatGroq(model= "openai/gpt-oss-20b",groq_api_key=GROQ_API_KEY )
model

# COMMAND ----------

#1 create prompt template

sys_template= "Translate the following into {language}"
prompt_template= ChatPromptTemplate.from_messages([
    ("system", sys_template),
    ("user", "{text}")
])

parser=StrOutputParser()

#create chain
chain= prompt_template | model | parser

#App Definition
app= FastAPI(title='langchain Serve pract',
             version=1.0,
             description='this is first langchain serve practice model')
#Add routes
add_routes( app,
           chain,
           path='/chain',

)

#main function
if __name__== "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)                                           