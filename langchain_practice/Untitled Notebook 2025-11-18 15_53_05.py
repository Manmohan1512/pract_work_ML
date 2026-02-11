# Databricks notebook source
pip install -r requirements.txt  


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from langchain_core.messages import AIMessage,HumanMessage
from pprint import pprint

messages=[AIMessage(content=f"Please tell me how can I help",name="LLMModel")]
messages.append(HumanMessage(content=f"I want to learn coding",name="Man"))
messages.append(AIMessage(content=f"Which programming language you want to learn",name="LLMModel"))
messages.append(HumanMessage(content=f"I want to learn python programming language",name="Man"))

for message in messages:
    message.pretty_print()

# COMMAND ----------

from langchain_groq import ChatGroq
GROQ_API_KEY="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"
#Gemma2-9b-it
llm= ChatGroq(model= "openai/gpt-oss-20b",groq_api_key=GROQ_API_KEY )
result= llm.invoke(messages)
result.response_metadata

# COMMAND ----------

def add(a:int,b:int)-> int:
    """ Add a and b
    Args:
        a (int): first int
        b (int): second int

    Returns:
        int
    """
    return a+b

# COMMAND ----------

llm_with_tools=llm.bind_tools([add])

tool_call=llm_with_tools.invoke([HumanMessage(content=f"What is 2 plus 2",name="Man")])
tool_call.tool_calls

# COMMAND ----------

from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage

class State(TypedDict):
    message:list[AnyMessage]

# COMMAND ----------

from langgraph.graph.message import add_messages
from typing import Annotated
class State(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]

# COMMAND ----------

initial_messages=[AIMessage(content=f"Please tell me how can I help",name="LLMModel")]
initial_messages.append(HumanMessage(content=f"I want to learn coding",name="Man"))
initial_messages
ai_message=AIMessage(content=f"Which programming language you want to learn",name="LLMModel")
ai_message

# COMMAND ----------

add_messages(initial_messages,ai_message)

# COMMAND ----------

def llm_tool(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

# COMMAND ----------

tools=[add]

# COMMAND ----------

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

builder=StateGraph(State)

## Add nodes

builder.add_node("llm_tool",llm_tool)
builder.add_node("tools",ToolNode(tools))

## Add Edge
builder.add_edge(START,"llm_tool")
builder.add_conditional_edges(
    "llm_tool",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition
)
builder.add_edge("tools",END)


graph_builder = builder.compile()

# COMMAND ----------

messages=graph_builder.invoke({"messages":"What is 2 plus 22"})

for message in messages["messages"]:
    message.pretty_print()