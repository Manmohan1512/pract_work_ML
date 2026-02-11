# Databricks notebook source
pip install -r requirements.txt


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------



from langchain_groq import ChatGroq
GROQ_API_KEY="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"
#Gemma2-9b-it
model= ChatGroq(model= "openai/gpt-oss-20b",groq_api_key=GROQ_API_KEY )
resp= model.invoke('hello')
resp.content

# COMMAND ----------

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

class State(TypedDict):
    topic:str
    story:str
    improved_story:str
    final_story:str

def generate_story(state:State):
    msg=model.invoke(f'Write 1 liner story premise about {state["topic"]}')
    return {'story':msg.content}

def check_conflict(state:State):
    if '?' in state['story'] or '!' in state['story']:
        return 'Fail'
    return 'Pass'

def improved_story(state:State):
    msg=model.invoke(f'improve this story vivid details {state["story"]}')
    return {'improved_story':msg.content}

def polish_story(state:State):
    msg=model.invoke(f'Add an unexpected twist to this story premise {state["story"]}')
    return {'final_story':msg.content}


# COMMAND ----------

#Build the graph
graph = StateGraph(State)

graph.add_node('generate', generate_story)
graph.add_node('improved', improved_story)
graph.add_node('polish', polish_story)

graph.add_edge(START, 'generate')
graph.add_conditional_edges('generate', check_conflict, {'Pass':'improved', 'Fail':'generate'})
graph.add_edge('improved', 'polish')
graph.add_edge('polish', END)

compile_graph= graph.compile()

graph_image = compile_graph.get_graph().draw_mermaid_png()

display(Image(graph_image))


# COMMAND ----------

state={'topic':'Agentic AI system'}
result= compile_graph.invoke(state)
result