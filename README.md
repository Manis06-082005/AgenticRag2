# AgenticRag

When the indexed documents do not provide enough evidence after the configured
retrieval retries, the agent searches the web with Tavily and grades the
results before using them to answer. Set `TAVILY_API_KEY` in the environment
to enable this fallback. Web search is only invoked when document retrieval is
insufficient.