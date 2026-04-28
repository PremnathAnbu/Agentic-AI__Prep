from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
import os
from dotenv import load_dotenv


os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    
model=ChatGroq(model="openai/gpt-oss-120b",temperature=0)

def graph():
    """Make a simple LLM agent"""
    graph_workflow=StateGraph(State)
    def call_model(state):
        return {"messages":[model.invoke(state['messages'])]}
    
    graph_workflow.add_node("agent",call_model)
    
    graph_workflow.add_edge(START,"agent")
    graph_workflow.add_edge("agent",END)
    
    agent=graph_workflow.compile()
    
    return agent



# from langchain_core.messages import SystemMessage

# def alternative_graph():

#     @tool
#     def add(a: float, b: float):
#         """Add two numbers"""
#         return a + b

#     tools = [add]
#     tool_node = ToolNode(tools)

#     model_with_tools = model.bind_tools(tools)

#     # ✅ Add system message
#     system_msg = SystemMessage(content="""
# You are a helpful AI assistant.

# Use tools when needed:
# - Use calculator tool for math
# - Do not guess calculations

# Return only final answer.
# """)

#     def call_model(state: State):
#         messages = state["messages"]

#         # inject system message only once
#         if not any(isinstance(m, SystemMessage) for m in messages):
#             messages = [system_msg] + messages

#         response = model_with_tools.invoke(messages)

#         return {"messages": [response]}

#     def should_continue(state: State):
#         last_message = state["messages"][-1]

#         if hasattr(last_message, "tool_calls") and last_message.tool_calls:
#             return "tools"

#         return END

#     graph_workflow = StateGraph(State)

#     graph_workflow.add_node("agent", call_model)
#     graph_workflow.add_node("tools", tool_node)

#     graph_workflow.add_edge(START, "agent")

#     # ✅ Conditional routing
#     graph_workflow.add_conditional_edges("agent", should_continue)

#     # ✅ Tool → back to agent (ReAct loop)
#     graph_workflow.add_edge("tools", "agent")

#     agent = graph_workflow.compile()

#     return agent

# agent=alternative_graph()


def alternative_graph():
    """Make a tool-calling agent"""
    
    @tool
    def add(a:float,b:float):
        """Add two numbers"""
        return a+b
    
    tool_node=ToolNode([add])
    model_with_tools=model.bind_tools([add])
    
    def call_model(state):
        return {"messages":[model_with_tools.invoke(state["messages"])]}
    
    def should_continue(state: State):
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return END
        
    graph_workflow=StateGraph(State)
    
    graph_workflow.add_node("agent",call_model)
    graph_workflow.add_node("tools",tool_node)
    
    graph_workflow.add_edge("tools","agent")
    graph_workflow.add_edge(START,"agent")
    graph_workflow.add_conditional_edges("agent",should_continue,{
        "tools": "tools",
        END: END
    })
    
    agent=graph_workflow.compile()
    return agent

agent=alternative_graph()
            
    


