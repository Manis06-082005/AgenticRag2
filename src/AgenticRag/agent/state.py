from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    search_query: str
    documents: list[Document]
    answer: str
    retries: int
    context_sufficient: bool
