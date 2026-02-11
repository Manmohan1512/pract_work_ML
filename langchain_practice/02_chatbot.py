# Databricks notebook source
pip install -r requirements.txt


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from typing import Annotated
from langgraph.graph.message import add_messages


# COMMAND ----------

class State(TypedDict):
    messages:Annotated[list, add_messages]

# COMMAND ----------

from langchain_groq import ChatGroq
GROQ_API_KEY="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"
#Gemma2-9b-it
model= ChatGroq(model= "openai/gpt-oss-20b",groq_api_key=GROQ_API_KEY )
resp= model.invoke('hello')
resp.content

# COMMAND ----------

# MAGIC %md
# MAGIC start with node in chatbot

# COMMAND ----------

#starting node in chat bot
def superbot(state:State):
    return {"messages":model.invoke(state["messages"])}

# COMMAND ----------


graph=StateGraph(State)
#node
graph.add_node('superbot',superbot)

#add edge
graph.add_edge(START,'superbot')
graph.add_edge('superbot',END)

graph_builder= graph.compile()

#display graph
from IPython.display import Image,display
display(Image(graph_builder.get_graph().draw_mermaid_png()))

# COMMAND ----------

graph_builder.invoke({"messages":"hello, I am manmohan and I am data engineer trying to learn AI"})

# COMMAND ----------

graph_builder.invoke({"messages":"hello, what is my name"})

# COMMAND ----------

#streaming the response
for event in graph_builder.stream({"messages":"hello, I am manmohan and I am data engineer trying to learn AI"}):
    print(event)

# COMMAND ----------

# MAGIC %md
# MAGIC State schema with Data class
# MAGIC

# COMMAND ----------

from typing import Literal
#from typing_extensions import TypeDict

class TypedDictState(TypedDict):
  name:str
  game:Literal['cricket','badminton']


def playgame(state:TypedDictState):
  print('playgame node has been called')
  return {'name':state['name'] + ' want to play'}


def cricket(state:TypedDictState):
  print('cricket node has been called')
  return {'name': state['name'] + ' cricket', 'game': 'cricket'}

def badminton(state:TypedDictState):
  print('badminton node has been called')
  return { 'name': state['name'] + ' badminton','game': 'badminton'}

import random
def decide_play(state:TypedDict)->Literal['cricket','badminton']:
  if random.random()< 0.5:
    return 'cricket'
  else:
    return 'badminton'



# COMMAND ----------

from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

builder= StateGraph(TypedDictState)
builder.add_node('playgame',playgame)
builder.add_node('cricket',cricket)
builder.add_node('badminton',badminton)

builder.add_edge(START,'playgame')
builder.add_conditional_edges('playgame',decide_play)
builder.add_edge('cricket',END)
builder.add_edge('badminton',END)
graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

# COMMAND ----------

graph.invoke({'name':'manmohan'})