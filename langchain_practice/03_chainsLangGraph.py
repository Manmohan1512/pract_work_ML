# Databricks notebook source
# MAGIC %md
# MAGIC Every message should have 3 main components:
# MAGIC - Content: content of messages
# MAGIC - name: specific name of author
# MAGIC - response_message: optional dict of meta data

# COMMAND ----------

pip install -r requirements.txt


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from typing import Annotated
from langgraph.graph.message import add_messages


# COMMAND ----------


from langchain_core.messages import HumanMessage, AIMessage
from pprint import pprint

messages= [AIMessage(content=f"Please tell me how are you", name='LLMModel')]
messages.append(HumanMessage(content=f"I want to learn more in AI coding", name='Man'))
messages.append(AIMessage(content=f"which programing language you want to learn", name='LLMModel'))
messages.append(HumanMessage(content=f"I want to learn python", name='Man'))

for msg in messages:
    msg.pretty_print()

# COMMAND ----------

# MAGIC %md
# MAGIC Chat Model

# COMMAND ----------

from langchain_groq import ChatGroq
GROQ_API_KEY="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"
#Gemma2-9b-it
model= ChatGroq(model= "openai/gpt-oss-20b",groq_api_key=GROQ_API_KEY )
resp= model.invoke(messages)
resp.content

# COMMAND ----------

# MAGIC %md
# MAGIC Tools
# MAGIC

# COMMAND ----------

def add(a: int, b: int, c :int=0) -> int:
    """ Add a and b
        Args:
        a (int): first int
        b (int): second int
        c(int): third int

    Returns:
        int
    """

    return a + b + c
    

# COMMAND ----------

#binding tool with LLM
model_with_tools = model.bind_tools([add])
#llm with tools
model_with_tools.invoke([HumanMessage(content= "what is the sum of 3 and 5?", name='man')])


# COMMAND ----------

# using messages as state
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import AnyMessage
class State(TypedDict):
    messages:list[AnyMessage]

def llm_tool(state:State):
    return {"messages": [model_with_tools.invoke(state["messages"])]}


# COMMAND ----------

from langgraph.graph.message import add_messages
from typing import Annotated
class State(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]

# COMMAND ----------

from langgraph.graph import StateGraph, START, END

graph=StateGraph(State)

#node
graph.add_node('llm_tool',llm_tool)

#add edge
graph.add_edge(START,'llm_tool')
graph.add_edge('llm_tool',END)

graph_builder= graph.compile()

#display graph
from IPython.display import Image,display
display(Image(graph_builder.get_graph().draw_mermaid_png()))

# COMMAND ----------

messages=graph_builder.invoke({"messages":"hello, what is 2 plus 5"})

for msg in messages['messages']:
    msg.pretty_print()

# COMMAND ----------

tools=[add]

# COMMAND ----------

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

builder=StateGraph(State)

#node
builder.add_node('llm_tool',llm_tool)
builder.add_node('tools',ToolNode(tools))

#add edge
builder.add_edge(START,'llm_tool')
builder.add_conditional_edges(
    'llm_tool',
      # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition)
builder.add_edge('tools',END)

graph_builder2= builder.compile()

#display graph
#from IPython.display import Image,display
#display(Image(builder2.get_graph().draw_mermaid_png()))

# COMMAND ----------



messages=graph_builder2.invoke({"messages":"what is  95, 85 "})

for message in messages["messages"]:
    message.pretty_print()
