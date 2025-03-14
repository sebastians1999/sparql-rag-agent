from typing import Annotated, Sequence, TypedDict, Callable, Dict, Any, Optional
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from scr.agent.utils.nodes.question_understanding import question_understanding
from scr.agent.utils.nodes.entity_resolution import retrieve_documents
from scr.agent.utils.nodes.sparql_query_construction import query_generator
from scr.agent.utils.state.state import State
from scr.agent.utils.config import Configuration


def create_graph(config: Optional[Configuration] = None) -> Graph:
    """Create the agent workflow graph.
    
    Args:
        config: Configuration for the runner. If None, a default config will be created.
        
    Returns:
        The compiled graph
    """
    # Create default config if none provided
    if config is None:
        config = Configuration(test_mode=True)
    
    # Create a new graph
    workflow = StateGraph(State) 

    # Add nodes with wrapper functions
    workflow.add_node("question_understanding", question_understanding)
    workflow.add_node("entity_resolution", retrieve_documents)
    workflow.add_node("sparql_query_construction", query_generator)

    # Define edges
    workflow.add_edge("question_understanding", "entity_resolution")
    workflow.add_edge("entity_resolution", "sparql_query_construction")

    # Set entry point
    workflow.set_entry_point("question_understanding")

    # Set exit point
    workflow.set_finish_point("sparql_query_construction")

    # Compile the graph
    graph = workflow.compile()

    return graph 