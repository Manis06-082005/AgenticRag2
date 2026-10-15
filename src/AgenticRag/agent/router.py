from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .state import AgentState


class RouteDecision(BaseModel):
    use_retrieval: bool = Field(
        description=(
            "Whether answering the question requires information from the "
            "user's indexed documents"
        )
    )


def route_question(
    state: AgentState,
    llm: BaseChatModel,
) -> Literal["rag", "conversation"]:
    classifier = llm.with_structured_output(RouteDecision)
    decision = classifier.invoke(
        [
            SystemMessage(
                content=(
                    "Decide whether the user's question needs information "
                    "from their indexed documents. Choose retrieval for "
                    "questions about the contents of their files, documents, "
                    "or provided materials. Choose normal conversation for "
                    "greetings, pleasantries, and general questions that do "
                    "not depend on those materials. Do not answer the question."
                )
            ),
            HumanMessage(content=state["question"]),
        ]
    )
    return "rag" if decision.use_retrieval else "conversation"


def route_after_context_check(
    state: AgentState,
    max_retries: int,
) -> Literal["generate", "rewrite_query", "search_web"]:
    if state.get("context_sufficient", False):
        return "generate"
    if state.get("retries", 0) < max_retries:
        return "rewrite_query"
    return "search_web"


def route_after_web_grade(
    state: AgentState,
) -> Literal["generate", "insufficient_context"]:
    if state.get("web_context_sufficient", False):
        return "generate"
    return "insufficient_context"