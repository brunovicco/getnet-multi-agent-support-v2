"""Application-level errors raised across ports."""


class LLMUnavailableError(Exception):
    """The LLM provider could not produce a response in time."""


class WebSearchUnavailableError(Exception):
    """The web search provider could not produce results in time."""
