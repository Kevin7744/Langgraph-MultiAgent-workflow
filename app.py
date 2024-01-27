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


# Create Agents
"""
The helper functions will help create agents. These agents will then be nodes in the graph
"""
import json

from langchain_core.messages import (
    AIMessage,
    BaseMessage, 
    ChatMessage,
    FunctionMessage, 
    HumanMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function

def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    functions = [format_tool_to_openai_function(t) for t in tools]
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                "Use the provided tools to progress towards answering the question."
                "If you are unable to fully answer, that's OK, another assistant with different tools"
                "will help where you left off. Execute what you can to make progress."
                "If you or any of the other assistants have the final answer or deliverables,"
                "prefix your response with FINAL ANSWER so the team know to stop."
                "You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="message"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names = ", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_functions(functions)

# Define tools
"""
Define the tools that our agents will use in the future
"""
from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_tool = TavilySearchResults(max_results=5)

# executes code locally

repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart"]
):
    """
    Use this to execute python code. If you want to see the output of a value, you should print it with
    'print(...) '. This is visible to the user.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n '''python\n{code}\n'''\nStdout: {result}"


# Create graph
"""
Create individual agents and tell them how to talk to each other using LangGraph
"""
# Define State
"""
Define the state of the graph. This will just a list of messages, along with a key to track the most recent sender
"""
import operator
from typing import List, Sequence, Tuple, TypedDict, Union

from langchain.agents import create_openai_functions_agent

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict


# This defines the object that is passed between each node
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

# Define agent nodes
import functools

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # Convert the agent output into a format that is suitable to append to a global state
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name,
    }

llm = ChatOpenAI(model="gpt-4-1106-preview")

# Research Agent and node
research_agent = create_agent(
    llm, 
    [tavily_tool],
    system_message="You should provide accurate data for the chart generator to use."
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# Chart generator
chart_agent = create_agent(
    llm, 
    [python_repl],
    system_message="Any charts you display will be visible by the user"
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="Chart Generator")