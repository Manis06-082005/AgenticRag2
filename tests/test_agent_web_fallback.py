import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from src.AgenticRag.agent.nodes import AgentNodes
from src.AgenticRag.agent.router import (
    route_after_context_check,
    route_after_web_grade,
)


class WebFallbackTests(unittest.TestCase):
    def setUp(self):
        self.nodes = object.__new__(AgentNodes)
        self.nodes.web_search = None
        self.nodes.web_max_results = 5

    def test_search_web_converts_results_to_cited_documents(self):
        result = {
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "content": "  Relevant evidence.  ",
                },
                {"url": "https://empty.example", "content": "  "},
            ]
        }
        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}),
            patch("src.AgenticRag.agent.nodes.TavilySearch") as tavily,
        ):
            tavily.return_value.invoke.return_value = result
            state = self.nodes.search_web(
                {"question": "Original", "search_query": "Rewritten"}
            )

        tavily.assert_called_once_with(max_results=5)
        tavily.return_value.invoke.assert_called_once_with(
            {"query": "Rewritten"}
        )
        self.assertEqual(len(state["documents"]), 1)
        document = state["documents"][0]
        self.assertEqual(document.page_content, "Relevant evidence.")
        self.assertEqual(document.metadata["source"], "https://example.com")
        self.assertEqual(document.metadata["source_type"], "web")
        self.assertFalse(state["web_context_sufficient"])

    def test_search_web_requires_api_key(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("src.AgenticRag.agent.nodes.TavilySearch") as tavily,
        ):
            with self.assertRaisesRegex(ValueError, "TAVILY_API_KEY"):
                self.nodes.search_web({"question": "Question"})
        tavily.assert_not_called()

    def test_grade_web_context_skips_model_for_empty_results(self):
        state = self.nodes.grade_web_context(
            {"question": "Question", "documents": []}
        )
        self.assertEqual(state, {"web_context_sufficient": False})

    def test_grade_web_context_checks_results_with_source_attribution(self):
        class FakeChecker:
            def invoke(self, messages):
                self.messages = messages
                return SimpleNamespace(sufficient=True)

        checker = FakeChecker()
        self.nodes.context_checker = checker
        state = self.nodes.grade_web_context(
            {
                "question": "Question",
                "documents": [
                    Document(
                        page_content="Evidence",
                        metadata={
                            "title": "Example",
                            "source": "https://example.com",
                        },
                    )
                ],
            }
        )

        self.assertTrue(state["web_context_sufficient"])
        self.assertIn("https://example.com", checker.messages[1].content)

    def test_router_uses_web_fallback_after_local_retries(self):
        self.assertEqual(
            route_after_context_check(
                {"context_sufficient": False, "retries": 2}, max_retries=2
            ),
            "search_web",
        )
        self.assertEqual(
            route_after_web_grade({"web_context_sufficient": True}),
            "generate",
        )
        self.assertEqual(
            route_after_web_grade({"web_context_sufficient": False}),
            "insufficient_context",
        )


if __name__ == "__main__":
    unittest.main()
