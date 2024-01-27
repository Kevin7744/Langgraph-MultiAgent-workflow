import getpass
import os

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass(f"Please provide your {var}")

_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

# add tracing in langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"


# Create tools
"""
One agent will do a web search with a search engine, and one agent to create plots.
"""
from typing import Annotated, List, Tuple, Union, TypedDict, Sequence

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool

tavily_tool = TavilySearchResults(max_results=5)

# Executes code locally
python_repl_tool = PythonREPLTool

# Helper Utilities
"""Defining a helper function makes it easy to add new agent worker nodes"""
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_agent(
        llm: ChatOpenAI, tools: list, system_prompt: str
):
    # Each worker node is given some names and some tools
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="Agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


# Create agent supervisor
"""It will use function to choose the next worker node or finish processing"""
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

members = ["Reseacher", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    "following workers: {members}. Given the following user request,"
    "respond with the worker to act next. Each worker will perfom a "
    "task and respond with their results and status. When finished,"
    "respond with FINISH."
)
# team supervisor is an LLM node, it just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members
# Using openai function callling can make output parsing easier
function_def = {
    "name": "route",
    "description": "Select the next role",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
} 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should work next?"
            "Or should we FINISH? Select one of : {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model="gpt-4-1106-preview")

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)


# Construct Graph
import operator
import functools

from langgraph.graph import StateGraph, END

# Agent state is the input to each node in the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

research_agent = create_agent(llm, [tavily_tool], "You are web researcher.")
research_node = functools.partial(agent_node, agent=research_agent, name="Reseacher")

# Performs code execution
code_agent = create_agent(llm, [python_repl_tool], "You may generate safe python code to anaylze data and generate charts using matplotlib.")
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

workflow = StateGraph(AgentState)
workflow.add_node("Reseacher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("Supervisor", supervisor_chain)

# Connect all the edges in the graph
for member in members:
    # Worker reports to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supevisor")
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# entry point
workflow.set_entry_point("supervisor")
graph = workflow.compile()


# Invoke the team
for s in graph.stream(
    {
        "messages": [
            HumanMessage(content="Code hello world and print it to the terminal")
        ]
    }
):
    if "__end__" not in s:
        print(s)
        print("----")