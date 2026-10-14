from pathlib import Path

from langgraph.graph import END, START, StateGraph

from .nodes import AgentNodes
from .router import route_after_context_check, route_question
from .state import AgentState
from ..utils import read_yaml


def build_graph(config_path: str | Path = "config/config.yaml"):
    config = read_yaml(str(config_path))
    max_retries = config.get("agent", {}).get("max_retries", 2)
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("agent.max_retries must be a non-negative integer.")

    nodes = AgentNodes(config_path)
    graph = StateGraph(AgentState)

    graph.add_node("normal_conversation", nodes.normal_conversation)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("rerank", nodes.rerank)
    graph.add_node("check_context", nodes.check_context)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("generate", nodes.generate)
    graph.add_node("insufficient_context", nodes.insufficient_context)

    graph.add_conditional_edges(
        START,
        lambda state: route_question(state, nodes.llm),
        {
            "conversation": "normal_conversation",
            "rag": "retrieve",
        },
    )
    graph.add_edge("normal_conversation", END)
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "check_context")
    graph.add_conditional_edges(
        "check_context",
        lambda state: route_after_context_check(state, max_retries),
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "insufficient_context": "insufficient_context",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("insufficient_context", END)

    return graph.compile()