from typing import NotRequired, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    search_query: NotRequired[str]
    documents: NotRequired[list[Document]]
    answer: NotRequired[str]
    retries: NotRequired[int]
    context_sufficient: NotRequired[bool]
